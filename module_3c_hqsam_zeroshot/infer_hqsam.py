import os
import random
import urllib.request
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# Zero-shot check for Rank #1 in docs/top_10_arc_segmentation_solutions.pdf:
# does Light HQ-SAM's pretrained boundary quality alone (no fine-tuning on
# our dataset) already beat fine-tuned MedSAM? Run this in its OWN fresh
# Kaggle session. "segment-anything-hq" (pip) installs under the same
# top-level "segment_anything" module name as the original "segment-anything"
# package module_3 uses - the two cannot coexist in one environment. Install
# ONLY "segment-anything-hq" here:
#   !pip uninstall -y -q segment-anything-hq segment-anything
#   !pip install -q "git+https://github.com/SysCV/sam-hq.git" ultralytics
# The PyPI package "segment-anything-hq" (v0.3) is a stale snapshot still
# using the plain (non-HQ) MaskDecoder - confirmed via a real
# `RuntimeError: Unexpected key(s)` on Kaggle (the checkpoint's HQ-specific
# keys like "mask_decoder.hf_token.weight" don't exist on that old decoder
# class). The current GitHub source uses MaskDecoderHQ, which matches the
# checkpoint - install from source, not PyPI.
#
# NOTE: "vit_tiny" (the literal "Light HQ-SAM" TinyViT/MobileSAM variant) is
# NOT registered in sam_model_registry from the pip package either -
# confirmed via a real KeyError on Kaggle (needs separate setup code; see
# github.com/SysCV/sam-hq/issues/130, unresolved upstream). Falling back to
# "vit_b" instead: same backbone size as our own MedSAM checkpoint, so this
# is still a fair "does HQ-SAM's boundary-focused training beat MedSAM's"
# test - just not literally the smallest edge variant, which only matters
# for the later distillation stage, not this check.
HQSAM_MODEL_TYPE = "vit_b"
HQSAM_CHECKPOINT_URL = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
HQSAM_CHECKPOINT_PATH = "/kaggle/working/sam_hq_vit_b.pth"

# Same trained detector from module_2b - localization already validated
# separately, only the segmenter changes in this experiment.
YOLO_CHECKPOINT_PATH = "/kaggle/input/datasets/hemantdarur/yolomodel-inference/yolo_checkpoints/tooth_yolo/weights/best.pt"

OUTPUT_DIR = "/kaggle/working/outputs_hqsam"
VAL_IMAGES_DIR = "/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4/val/images"

IMG_SIZE = 1024
NUM_SAMPLE_IMAGES = 15  # same sample size/seed as module_3's runs - directly comparable
RANDOM_SEED = 42

YOLO_CONF_THRESHOLD = 0.25
YOLO_BBOX_MARGIN = 20

# HQ-SAM's own output-blending toggle (see their demo): True uses only the
# HQ-refined token, recommended for a single isolated object; False blends
# with the standard SAM token, recommended for cluttered/multi-part scenes.
# Our mask covers a whole arch of many teeth - closer to "multi-part".
HQ_TOKEN_ONLY = False

# Same mask cleanup as module_3, for a fair before/after (not the interior
# enamel ROI - that's a module_5 color-reading concern, not this module's).
ENABLE_MASK_CLEANUP = True
CLEANUP_CLOSE_KERNEL_SIZE = 5
CLEANUP_MIN_COMPONENT_FRACTION = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------
def load_hqsam_model(checkpoint_path, checkpoint_url):
    if not os.path.exists(checkpoint_path):
        print(f"Downloading Light HQ-SAM checkpoint to {checkpoint_path} ...")
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)

    sam = sam_model_registry[HQSAM_MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(DEVICE)
    sam.eval()
    return SamPredictor(sam)


def load_yolo_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"YOLO checkpoint not found at {checkpoint_path} - "
              f"falling back to no localization for this run.")
        return None
    return YOLO(checkpoint_path)


# --------------------------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------------------------
def preprocess_image(image_path):
    # Same square resize convention as module_3, so the YOLO box prompt
    # (produced in this same 1024x1024 frame) and HQ-SAM's own prediction
    # both operate in one consistent coordinate space.
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_resized = resize(
        image_rgb, (IMG_SIZE, IMG_SIZE), order=3, preserve_range=True, anti_aliasing=True
    )
    image_resized_uint8 = image_resized.astype(np.uint8)
    return image_rgb, image_resized_uint8


# --------------------------------------------------------------------------
# BBOX PROMPT
# --------------------------------------------------------------------------
def get_bbox_prompt_yolo(image_resized_uint8, yolo_model, margin=YOLO_BBOX_MARGIN):
    results = yolo_model.predict(image_resized_uint8, conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if boxes.shape[0] == 0:
        return None

    x_min = boxes[:, 0].min()
    y_min = boxes[:, 1].min()
    x_max = boxes[:, 2].max()
    y_max = boxes[:, 3].max()

    h, w = image_resized_uint8.shape[:2]
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


# --------------------------------------------------------------------------
# MASK CLEANUP (same recipe as module_3's cleanup_mask)
# --------------------------------------------------------------------------
def cleanup_mask(mask, min_component_fraction=CLEANUP_MIN_COMPONENT_FRACTION,
                  close_kernel_size=CLEANUP_CLOSE_KERNEL_SIZE):
    kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return closed

    largest_area = stats[1:, cv2.CC_STAT_AREA].max()
    min_area = largest_area * min_component_fraction

    cleaned = np.zeros_like(closed)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 1
    return cleaned


# --------------------------------------------------------------------------
# INFERENCE
# --------------------------------------------------------------------------
def run_inference(predictor, image_resized_uint8, bbox):
    predictor.set_image(image_resized_uint8)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox,
        multimask_output=False,
        hq_token_only=HQ_TOKEN_ONLY,
    )
    mask = masks[0].astype(np.uint8)
    if ENABLE_MASK_CLEANUP:
        mask = cleanup_mask(mask)
    return mask


# --------------------------------------------------------------------------
# TOOTH CROP EXTRACTION
# --------------------------------------------------------------------------
def extract_tooth_crop(original_image, mask, save_path):
    mask_resized = cv2.resize(
        mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    mask_3ch = np.repeat(mask_resized[:, :, None], 3, axis=2)
    tooth_crop = original_image * mask_3ch

    cv2.imwrite(save_path, cv2.cvtColor(tooth_crop, cv2.COLOR_RGB2BGR))
    return tooth_crop


# --------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------
def visualize_result(original_image, mask, tooth_crop, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask (Light HQ-SAM, zero-shot)")
    axes[1].axis("off")

    axes[2].imshow(tooth_crop)
    axes[2].set_title("Tooth Crop")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    predictor = load_hqsam_model(HQSAM_CHECKPOINT_PATH, HQSAM_CHECKPOINT_URL)
    yolo_model = load_yolo_model(YOLO_CHECKPOINT_PATH)

    all_images = sorted(
        f for f in os.listdir(VAL_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg"))
    )
    sample_images = random.sample(all_images, min(NUM_SAMPLE_IMAGES, len(all_images)))

    for filename in sample_images:
        image_path = os.path.join(VAL_IMAGES_DIR, filename)
        base_name = os.path.splitext(filename)[0]

        original_image, image_resized_uint8 = preprocess_image(image_path)

        bbox = None
        if yolo_model is not None:
            bbox = get_bbox_prompt_yolo(image_resized_uint8, yolo_model)
        if bbox is None:
            print(f"  No YOLO detection for {filename} - skipping (no generic-box fallback "
                  f"in this zero-shot check, to isolate the segmenter comparison).")
            continue

        mask = run_inference(predictor, image_resized_uint8, bbox)

        crop_path = os.path.join(OUTPUT_DIR, f"{base_name}_tooth_crop.png")
        tooth_crop = extract_tooth_crop(original_image, mask, crop_path)

        viz_path = os.path.join(OUTPUT_DIR, f"{base_name}_hqsam_result.png")
        visualize_result(original_image, mask, tooth_crop, viz_path)

        print(f"Saved tooth crop for {filename} -> {crop_path}")
