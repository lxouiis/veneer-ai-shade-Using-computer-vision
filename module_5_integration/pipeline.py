import os
import sys
import json
import random
import numpy as np
import cv2

# module_3 and module_4 are sibling folders, not installed packages, so make
# them importable by path rather than duplicating their code here.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJECT_ROOT, "module_3_medsam_inference"))
sys.path.append(os.path.join(_PROJECT_ROOT, "module_4_bisenet"))

from infer import (  # noqa: E402
    load_model as load_medsam_model,
    load_yolo_model,
    preprocess_image as preprocess_for_medsam,
    get_bbox_prompt,
    get_bbox_prompt_yolo,
    run_inference as run_medsam_inference,
    extract_tooth_crop,
    get_interior_enamel_mask,
    CHECKPOINT_PATH as MEDSAM_CHECKPOINT_PATH,
    YOLO_CHECKPOINT_PATH,  # module_3's ORIGINAL detector - not the smile-finetuned
                           # one, which evaluate_new_yolo.py showed was a net wash
                           # (higher skip rate, lower median DSC) despite fixing a
                           # few specific odd-angle cases
)
from parse_face import (  # noqa: E402
    load_bisenet_model,
    preprocess_for_bisenet,
    run_face_parsing,
    extract_region_crop,
    SKIN_LABELS,
    LIP_LABELS,
    BISENET_WEIGHTS,
)

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
OUTPUT_DIR = "/kaggle/working/pipeline_outputs"

# The real combined NCT dataset, not the earlier CelebAMask-HQ smoke-test
# placeholder. BiSeNet still needs a face in frame - many NCT photos are
# tight intraoral close-ups with no visible skin/lip, which is expected and
# handled by the graceful-degradation flags below, not a bug to fix here.
SAMPLE_IMAGES_DIR = "/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4/train/images"
NUM_SAMPLE_IMAGES = 700
RANDOM_SEED = 42

# Doc Sans's human-verified masks (doctor_review_app's approved_masks folder,
# synced down from Drive - see doctor_review_app/download_approved_masks.py).
# When a verified mask exists for an image, it's used for the tooth crop
# instead of a fresh MedSAM prediction - a human-checked mask is a strictly
# more trustworthy color-extraction source, which matters a lot if this
# output ends up as pseudo-label training data (module_6_vit_clip). None
# means "skip this - run raw inference for everything" (e.g. before any
# review has happened yet).
APPROVED_MASKS_DIR = "/kaggle/input/datasets/hemantdarur/doctor-approved-masks/approved_masks"

COLOR_VECTOR_LABELS = [
    "tooth_L", "tooth_a", "tooth_b",
    "skin_L", "skin_a", "skin_b",
    "lip_L", "lip_a", "lip_b",
]


# --------------------------------------------------------------------------
# CIELAB EXTRACTION
# --------------------------------------------------------------------------
def extract_cielab(crop_image, mask=None):
    # All crops in this pipeline are RGB (module_3 and module_4 both convert
    # to RGB early), so we convert RGB -> LAB here, not BGR -> LAB.
    if crop_image is None:
        return float("nan"), float("nan"), float("nan")

    if mask is None:
        # Fall back to "non-black pixel" as the foreground, since our crops
        # have the background zeroed out by masking upstream.
        mask = np.any(crop_image != 0, axis=2)
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    lab = cv2.cvtColor(crop_image, cv2.COLOR_RGB2LAB).astype(np.float32)

    # OpenCV encodes 8-bit LAB as L:0-255, a/b:0-255 (offset by 128). Rescale
    # to true CIELab ranges (L:0-100, a/b: roughly -128 to 127) before
    # averaging, only over the foreground (masked) pixels.
    l_mean = lab[..., 0][mask_bool].mean() * (100.0 / 255.0)
    a_mean = lab[..., 1][mask_bool].mean() - 128.0
    b_mean = lab[..., 2][mask_bool].mean() - 128.0

    return float(l_mean), float(a_mean), float(b_mean)


# --------------------------------------------------------------------------
# METADATA
# --------------------------------------------------------------------------
def save_meta(output_dir, patient_id, color_vector, bbox, mask_areas, flags):
    meta = {
        "patient_id": patient_id,
        "color_vector": color_vector,
        "color_vector_labels": COLOR_VECTOR_LABELS,
        "bbox": list(bbox),
        "mask_areas": mask_areas,
        "flags": flags,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


# --------------------------------------------------------------------------
# STAGE 1 PIPELINE: MEDSAM -> BISENET -> CIELAB -> SAVE
# --------------------------------------------------------------------------
def run_stage1_pipeline(image_path, medsam_model, yolo_model, bisenet_model, output_dir,
                         approved_masks_dir=None):
    patient_id = os.path.splitext(os.path.basename(image_path))[0]
    patient_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    flags = []

    # preprocess_image returns 3 values (tensor, full-res RGB, 1024x1024
    # uint8 for YOLO) - previously only 2 were unpacked here, which would
    # have raised ValueError the first time this actually ran.
    medsam_tensor, medsam_original, medsam_resized_uint8 = preprocess_for_medsam(image_path)

    # ---- Step 1: tooth mask - prefer Doc Sans's human-verified mask over a
    # fresh model prediction, when one exists. A doctor-checked mask is a
    # strictly more trustworthy color-extraction source than raw model
    # output, which matters most if this ends up feeding pseudo-label
    # training data (module_6_vit_clip) rather than just a one-off reading.
    approved_mask_path = None
    if approved_masks_dir:
        candidate = os.path.join(approved_masks_dir, f"{patient_id}_mask.png")
        if os.path.exists(candidate):
            approved_mask_path = candidate

    if approved_mask_path:
        raw_mask = cv2.imread(approved_mask_path, cv2.IMREAD_GRAYSCALE)
        # doctor_review_app's hand-edit tool saves 4 distinct pixel values,
        # not a plain boolean - 255=tooth, 170=ignore (overlap/ambiguous,
        # excluded here so it can't contaminate the color reading), 85=
        # anomaly (flagged, also excluded from today's color math, but its
        # presence is recorded below rather than silently discarded - it's
        # meant for a future training stage, not thrown away). A raw,
        # un-edited approval only ever contains 0/255 (see app.py's "Looks
        # good" path), so this still works correctly for those too.
        tooth_mask = (raw_mask == 255).astype(np.uint8)
        if np.any(raw_mask == 170):
            flags.append("ignore_region_present")
        if np.any(raw_mask == 85):
            flags.append("anomaly_region_present")
        flags.append("tooth_mask_doctor_approved")
        # No model box prompt was used for an approved mask - record the
        # mask's own extent instead, so meta.json's bbox field still means
        # "where the tooth is" consistently, not "what box MedSAM was given."
        ys, xs = np.where(tooth_mask > 0)
        bbox = (np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
                if xs.size > 0 else np.array([0, 0, 0, 0], dtype=np.float32))
    else:
        bbox = get_bbox_prompt_yolo(medsam_resized_uint8, yolo_model) if yolo_model is not None else None
        if bbox is None:
            bbox = get_bbox_prompt(medsam_tensor.shape[-2:])
        tooth_mask = run_medsam_inference(medsam_model, medsam_tensor, bbox)
        flags.append("tooth_mask_model_predicted")

    tooth_crop = extract_tooth_crop(medsam_original, tooth_mask, os.path.join(patient_dir, "tooth.png"))
    tooth_fg_mask = np.any(tooth_crop != 0, axis=2)
    if tooth_fg_mask.sum() == 0:
        flags.append("tooth_region_not_found")

    # Adaptive Interior Enamel ROI: the tooth-gum boundary is where color
    # contamination is worst even on an otherwise-good mask, so read color
    # only from each tooth blob's interior core rather than the full mask -
    # see module_3's get_interior_enamel_mask and docs/EXPERIMENTS.md.
    # Falls back to the full mask if erosion empties it out entirely (very
    # thin/small tooth region), rather than returning no color at all.
    tooth_color_mask = get_interior_enamel_mask(tooth_fg_mask)
    if tooth_color_mask.sum() == 0:
        tooth_color_mask = tooth_fg_mask
        flags.append("interior_enamel_roi_empty_fallback_full_mask")

    # ---- Step 2: BiSeNet skin + lip crops ----
    bisenet_tensor, bisenet_original = preprocess_for_bisenet(image_path)
    parsing_map = run_face_parsing(bisenet_model, bisenet_tensor)
    skin_crop, skin_mask = extract_region_crop(bisenet_original, parsing_map, SKIN_LABELS, "skin")
    lip_crop, lip_mask = extract_region_crop(bisenet_original, parsing_map, LIP_LABELS, "lip")

    if skin_crop is not None:
        cv2.imwrite(os.path.join(patient_dir, "skin.png"), cv2.cvtColor(skin_crop, cv2.COLOR_RGB2BGR))
    else:
        flags.append("skin_region_not_found")

    if lip_crop is not None:
        cv2.imwrite(os.path.join(patient_dir, "lip.png"), cv2.cvtColor(lip_crop, cv2.COLOR_RGB2BGR))
    else:
        flags.append("lip_region_not_found")

    # ---- Step 3: CIELab from all 3 crops (9 numbers total) ----
    tooth_lab = extract_cielab(tooth_crop, tooth_color_mask)
    skin_lab = extract_cielab(skin_crop, skin_mask)
    lip_lab = extract_cielab(lip_crop, lip_mask)
    color_vector = list(tooth_lab) + list(skin_lab) + list(lip_lab)

    # ---- Step 5: meta.json ----
    mask_areas = {
        "tooth_px": int(tooth_fg_mask.sum()),
        "tooth_interior_enamel_px": int(tooth_color_mask.sum()),
        "skin_px": int(skin_mask.sum()) if skin_mask is not None else 0,
        "lip_px": int(lip_mask.sum()) if lip_mask is not None else 0,
    }
    save_meta(patient_dir, patient_id, color_vector, bbox.tolist(), mask_areas, flags)

    return {
        "tooth_crop": tooth_crop,
        "skin_crop": skin_crop,
        "lip_crop": lip_crop,
        "color_vector": color_vector,
    }


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    medsam_model = load_medsam_model(MEDSAM_CHECKPOINT_PATH)
    yolo_model = load_yolo_model(YOLO_CHECKPOINT_PATH)
    bisenet_model = load_bisenet_model(BISENET_WEIGHTS)

    # None if the attached dataset isn't present yet (e.g. no review has
    # happened, or doctor_review_app hasn't been run this session) - every
    # image then just falls back to a fresh model prediction, same as before.
    approved_masks_dir = APPROVED_MASKS_DIR if os.path.exists(APPROVED_MASKS_DIR) else None
    if approved_masks_dir is None:
        print(f"No approved masks found at {APPROVED_MASKS_DIR} - "
              f"using fresh model predictions for every image this run.")

    all_images = sorted(
        f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    sample_images = random.sample(all_images, min(NUM_SAMPLE_IMAGES, len(all_images)))

    for filename in sample_images:
        image_path = os.path.join(SAMPLE_IMAGES_DIR, filename)
        result = run_stage1_pipeline(
            image_path, medsam_model, yolo_model, bisenet_model, OUTPUT_DIR,
            approved_masks_dir=approved_masks_dir,
        )
        print(f"{filename} -> color_vector ({', '.join(COLOR_VECTOR_LABELS)}):")
        print(f"  {result['color_vector']}")
