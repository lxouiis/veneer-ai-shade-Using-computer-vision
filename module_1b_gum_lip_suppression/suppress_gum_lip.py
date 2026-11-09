import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# A standalone, classical (no model) pre-filter: colour-based gum/lip
# detection, run BEFORE YOLO localization and BEFORE segmentation (module_2b,
# module_3/3c), not after. See notes.md for why this needs to sit this early
# rather than as another mask-cleanup step at the end.
OUTPUT_DIR = "/kaggle/working/outputs_gum_lip_suppression"
VAL_IMAGES_DIR = "/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4/val/images"

IMG_SIZE = 1024  # same working resolution as module_1/module_3, for a fair preview
NUM_SAMPLE_IMAGES = 15
RANDOM_SEED = 42

# Thresholds on OpenCV's Lab (true a = a_cv - 128) and HSV. Gum/lip tissue is
# reliably redder (higher a) and more saturated than enamel, which is close
# to neutral grey/cream even under warm or cool lighting - see notes.md for
# why 'a' is the primary channel and hue is deliberately not used alone.
GUM_LIP_A_THRESHOLD = 12      # true Lab 'a'; enamel sits close to 0, gum/lip well above this
GUM_LIP_SAT_THRESHOLD = 40    # HSV saturation (0-255); filters near-grey shadow noise
GUM_LIP_VALUE_FLOOR = 25      # HSV value (0-255); ignores near-black gaps between teeth

# Grows the detected gum/lip region outward by this many pixels before
# burning it out, to also catch the blended transition band right at the
# tooth-gum edge (the boundary color-leakage failure noted in
# module_3/notes.md and docs/EXPERIMENTS.md), not just pure-gum pixels.
GUM_LIP_DILATE_PX = 6

# Same cleanup recipe as module_2b/3/3c's cleanup_mask, applied to the
# gum/lip mask itself so a few stray mis-classified pixels don't create
# speckled holes in the burned-out image.
CLEANUP_CLOSE_KERNEL_SIZE = 5
CLEANUP_MIN_COMPONENT_FRACTION = 0.01

# What burned-out pixels get set to. Black matches the convention used for
# "removed" background everywhere else in this project (module_3/5's crops),
# and gives YOLO/SAM the strongest possible signal to look elsewhere.
BURN_COLOR = (0, 0, 0)


# --------------------------------------------------------------------------
# GUM/LIP COLOUR MASK
# --------------------------------------------------------------------------
def get_gum_lip_mask(image_rgb):
    """Binary mask (1 = gum/lip tissue) from colour alone - no model, no
    training data, so it can run on any framing (tight intraoral close-up or
    a fuller smile-with-chin shot) before anything else in the pipeline."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    true_a = lab[:, :, 1].astype(np.int16) - 128  # OpenCV shifts 'a' by 128 to fit uint8

    mask = (
        (true_a > GUM_LIP_A_THRESHOLD)
        & (saturation > GUM_LIP_SAT_THRESHOLD)
        & (value > GUM_LIP_VALUE_FLOOR)
    ).astype(np.uint8)

    mask = _cleanup_mask(mask)

    if GUM_LIP_DILATE_PX > 0:
        kernel = np.ones((GUM_LIP_DILATE_PX, GUM_LIP_DILATE_PX), np.uint8)
        mask = cv2.dilate(mask, kernel)

    return mask


def _cleanup_mask(mask, min_component_fraction=CLEANUP_MIN_COMPONENT_FRACTION,
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
# BURN-OUT
# --------------------------------------------------------------------------
def burn_out_gum_lip(image_rgb, gum_lip_mask, burn_color=BURN_COLOR):
    """Returns a copy of image_rgb with every gum/lip pixel replaced by
    burn_color. Run this BEFORE the image reaches YOLO or the segmenter -
    see notes.md for why post-hoc masking isn't equivalent."""
    burned = image_rgb.copy()
    burned[gum_lip_mask.astype(bool)] = burn_color
    return burned


# --------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------
def visualize_suppression(image_rgb, gum_lip_mask, burned_image, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gum_lip_mask, cmap="gray")
    axes[1].set_title("Detected Gum/Lip (colour-based)")
    axes[1].axis("off")

    axes[2].imshow(burned_image)
    axes[2].set_title("Burned Out (feed this to YOLO/SAM)")
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

    all_images = sorted(
        f for f in os.listdir(VAL_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg"))
    )
    sample_images = random.sample(all_images, min(NUM_SAMPLE_IMAGES, len(all_images)))

    for filename in sample_images:
        image_path = os.path.join(VAL_IMAGES_DIR, filename)
        base_name = os.path.splitext(filename)[0]

        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        gum_lip_mask = get_gum_lip_mask(image_rgb)
        burned_image = burn_out_gum_lip(image_rgb, gum_lip_mask)

        viz_path = os.path.join(OUTPUT_DIR, f"{base_name}_gum_lip_suppression.png")
        visualize_suppression(image_rgb, gum_lip_mask, burned_image, viz_path)

        burned_percentage = 100.0 * gum_lip_mask.sum() / gum_lip_mask.size
        print(f"{filename}: burned {burned_percentage:.1f}% of pixels as gum/lip -> {viz_path}")
