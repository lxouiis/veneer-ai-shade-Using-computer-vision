import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from model import BiSeNet

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
BISENET_WEIGHTS = "/kaggle/working/bisenet/79999_iter.pth"
OUTPUT_DIR = "/kaggle/working/outputs"

# Face photos to run parsing on. BiSeNet expects a full face, unlike the NCT
# tooth close-ups used in modules 0-3. Using the CelebAMask-HQ Kaggle mirror
# (ipythonx/celebamaskhq) here is a good smoke test since BiSeNet was itself
# trained on this dataset. Confirmed path on Kaggle (2026-09-03) - this mirror
# nests everything under a datasets/ prefix plus a CelebAMask-HQ/ subfolder.
# Swap for your own patient photos later.
SAMPLE_IMAGES_DIR = "/kaggle/input/datasets/ipythonx/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img"
NUM_SAMPLE_IMAGES = 3

INPUT_SIZE = 512
N_CLASSES = 19
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RANDOM_SEED = 42

# CelebAMask-HQ 19-class label map, as used by zllrunning/face-parsing.PyTorch.
LABEL_MAP = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",   # eyeglasses
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",   # earring
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l", # necklace
    16: "cloth",
    17: "hair",
    18: "hat",
}
SKIN_LABELS = [1]
LIP_LABELS = [12, 13]  # upper lip + lower lip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------
def load_bisenet_model(weights_path):
    model = BiSeNet(n_classes=N_CLASSES)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# --------------------------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------------------------
def preprocess_for_bisenet(image_path):
    original_image = Image.open(image_path).convert("RGB")

    to_tensor = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image_tensor = to_tensor(original_image).unsqueeze(0)  # add batch dim -> (1, 3, 512, 512)

    return image_tensor, original_image


# --------------------------------------------------------------------------
# FACE PARSING
# --------------------------------------------------------------------------
def run_face_parsing(model, image_tensor):
    with torch.no_grad():
        out = model(image_tensor.to(DEVICE))[0]  # main head only; see model.py
    parsing_map = out.squeeze(0).argmax(dim=0).cpu().numpy()  # (512, 512) class-id map
    return parsing_map


# --------------------------------------------------------------------------
# REGION CROP EXTRACTION
# --------------------------------------------------------------------------
def extract_region_crop(original_image, parsing_map, labels, region_name):
    original_np = np.array(original_image)
    orig_h, orig_w = original_np.shape[:2]

    # parsing_map is 512x512; scale it back up to the original resolution with
    # nearest neighbor so we don't invent fractional class ids at edges.
    parsing_resized = cv2.resize(
        parsing_map.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )

    mask = np.isin(parsing_resized, labels).astype(np.uint8)
    if mask.sum() == 0:
        print(f"No pixels found for region '{region_name}' (labels={labels}).")
        return None, None

    mask_3ch = np.repeat(mask[:, :, None], 3, axis=2)
    crop = original_np * mask_3ch
    return crop, mask


# --------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------
def visualize_parsing(original_image, parsing_map, skin_crop, lip_crop, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(parsing_map, cmap="tab20")
    axes[1].set_title("Parsing Map (19 classes)")
    axes[1].axis("off")

    axes[2].imshow(skin_crop if skin_crop is not None else np.zeros_like(np.array(original_image)))
    axes[2].set_title("Skin Crop" if skin_crop is not None else "Skin (not found)")
    axes[2].axis("off")

    axes[3].imshow(lip_crop if lip_crop is not None else np.zeros_like(np.array(original_image)))
    axes[3].set_title("Lip Crop" if lip_crop is not None else "Lip (not found)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    model = load_bisenet_model(BISENET_WEIGHTS)

    all_images = sorted(
        f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    sample_images = random.sample(all_images, min(NUM_SAMPLE_IMAGES, len(all_images)))

    for filename in sample_images:
        image_path = os.path.join(SAMPLE_IMAGES_DIR, filename)
        base_name = os.path.splitext(filename)[0]

        image_tensor, original_image = preprocess_for_bisenet(image_path)
        parsing_map = run_face_parsing(model, image_tensor)

        skin_crop, skin_mask = extract_region_crop(original_image, parsing_map, SKIN_LABELS, "skin")
        lip_crop, lip_mask = extract_region_crop(original_image, parsing_map, LIP_LABELS, "lip")

        if skin_crop is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_skin_crop.png"),
                        cv2.cvtColor(skin_crop, cv2.COLOR_RGB2BGR))
        if lip_crop is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_lip_crop.png"),
                        cv2.cvtColor(lip_crop, cv2.COLOR_RGB2BGR))

        viz_path = os.path.join(OUTPUT_DIR, f"{base_name}_parsing_result.png")
        visualize_parsing(original_image, parsing_map, skin_crop, lip_crop, viz_path)

        print(f"Parsed {filename}: skin_found={skin_crop is not None}, lip_found={lip_crop is not None}")
