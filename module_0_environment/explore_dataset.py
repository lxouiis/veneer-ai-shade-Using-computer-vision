# explore_dataset.py
# Explores the NCT tooth dataset folder structure, verifies the environment,
# and displays sample images/masks/labels so we can sanity-check the data
# before building the rest of the pipeline.

import os
import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
DATASET_ROOT = "/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4"
OUTPUT_DIR = "/kaggle/working/module_0_environment"

TRAIN_IMAGES_DIR = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_MASKS_DIR = os.path.join(DATASET_ROOT, "train", "masks")
TRAIN_LABELS_DIR = os.path.join(DATASET_ROOT, "train", "labels")
VAL_IMAGES_DIR = os.path.join(DATASET_ROOT, "val", "images")
VAL_MASKS_DIR = os.path.join(DATASET_ROOT, "val", "masks")
VAL_LABELS_DIR = os.path.join(DATASET_ROOT, "val", "labels")

IMAGE_EXTENSIONS = (".jpg", ".jpeg")  # dataset has both extensions mixed in
MASK_SUFFIX = "_mask.png"
LABEL_EXT = ".txt"

MAX_FILES_TO_LIST = 3       # how many filenames to preview per folder in the walk
NUM_DISPLAY_SAMPLES = 3     # how many images/masks to show side by side
NUM_OVERLAY_SAMPLES = 2     # how many image+mask+overlay rows to show
MASK_BINARY_THRESHOLD = 127 # pixel value above which a mask pixel counts as "tooth"
OVERLAY_ALPHA = 0.4         # transparency of the red mask overlay
DIM_SAMPLE_SIZE = 50        # how many images to sample when computing average dimensions
RANDOM_SEED = 42            # fixed seed so sample selection is reproducible


# --------------------------------------------------------------------------
# SECTION 1: IMPORTS AND VERSION CHECK
# --------------------------------------------------------------------------
def section1_version_check():
    print("=" * 70)
    print("SECTION 1: IMPORTS AND VERSION CHECK")
    print("=" * 70)
    print(f"Python version   : {sys.version}")
    print(f"OpenCV version   : {cv2.__version__}")
    print(f"NumPy version    : {np.__version__}")
    print(f"Matplotlib ver.  : {plt.matplotlib.__version__}")
    print(f"scikit-image ver.: {skimage.__version__}")
    print(f"Dataset path exists: {os.path.exists(DATASET_ROOT)} -> {DATASET_ROOT}")
    print()


# --------------------------------------------------------------------------
# SECTION 2: FOLDER STRUCTURE
# --------------------------------------------------------------------------
def count_files(directory, ext):
    if not os.path.isdir(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(ext)])


def section2_folder_structure():
    print("=" * 70)
    print("SECTION 2: FOLDER STRUCTURE")
    print("=" * 70)

    for root, dirs, files in os.walk(DATASET_ROOT):
        dirs.sort()
        files = sorted(files)
        print(f"[DIR] {root}  ({len(files)} files)")
        for f in files[:MAX_FILES_TO_LIST]:
            print(f"    - {f}")

    print("\nFile counts:")
    for split, images_dir, masks_dir, labels_dir in [
        ("train", TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, TRAIN_LABELS_DIR),
        ("val", VAL_IMAGES_DIR, VAL_MASKS_DIR, VAL_LABELS_DIR),
    ]:
        n_images = count_files(images_dir, IMAGE_EXTENSIONS)
        n_masks = count_files(masks_dir, ".png")
        n_labels = count_files(labels_dir, LABEL_EXT)
        print(f"  {split}: images={n_images}, masks={n_masks}, labels={n_labels}")
    print()


# --------------------------------------------------------------------------
# SHARED HELPER: MATCH IMAGES TO MASKS
# --------------------------------------------------------------------------
def get_mask_filename(image_filename):
    base_name, _ = os.path.splitext(image_filename)
    return base_name + MASK_SUFFIX


def find_matching_pairs(images_dir, masks_dir):
    pairs = []
    if not os.path.isdir(images_dir):
        return pairs
    for image_filename in sorted(os.listdir(images_dir)):
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        mask_path = os.path.join(masks_dir, get_mask_filename(image_filename))
        if os.path.exists(mask_path):
            image_path = os.path.join(images_dir, image_filename)
            pairs.append((image_path, mask_path))
    return pairs


def sample_pairs(pairs, k, seed=RANDOM_SEED):
    rng = random.Random(seed)
    return rng.sample(pairs, min(k, len(pairs)))


# --------------------------------------------------------------------------
# SECTION 3: DISPLAY SAMPLE IMAGES
# --------------------------------------------------------------------------
def section3_display_images(sample):
    print("=" * 70)
    print("SECTION 3: DISPLAY SAMPLE IMAGES")
    print("=" * 70)

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    for ax, (image_path, _) in zip(axes, sample):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(os.path.basename(image_path), fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))
    plt.show()
    print()


# --------------------------------------------------------------------------
# SECTION 4: DISPLAY SAMPLE MASKS
# --------------------------------------------------------------------------
def section4_display_masks(sample):
    print("=" * 70)
    print("SECTION 4: DISPLAY SAMPLE MASKS")
    print("=" * 70)

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    for ax, (_, mask_path) in zip(axes, sample):
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(mask_gray, cmap="gray")
        ax.set_title(os.path.basename(mask_path), fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_masks.png"))
    plt.show()
    print()


# --------------------------------------------------------------------------
# SECTION 5: OVERLAY CHECK
# --------------------------------------------------------------------------
def section5_overlay_check(overlay_sample):
    print("=" * 70)
    print("SECTION 5: OVERLAY CHECK")
    print("=" * 70)

    fig, axes = plt.subplots(len(overlay_sample), 3, figsize=(12, 4 * len(overlay_sample)))
    if len(overlay_sample) == 1:
        axes = [axes]

    for row, (image_path, mask_path) in zip(axes, overlay_sample):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask_gray > MASK_BINARY_THRESHOLD).astype(np.uint8)

        row[0].imshow(image_rgb)
        row[0].set_title("Image")
        row[0].axis("off")

        row[1].imshow(mask_binary, cmap="gray")
        row[1].set_title("Mask")
        row[1].axis("off")

        row[2].imshow(image_rgb)
        row[2].imshow(mask_binary, cmap="Reds", alpha=OVERLAY_ALPHA)
        row[2].set_title("Overlay")
        row[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overlay_check.png"))
    plt.show()
    print()


# --------------------------------------------------------------------------
# SECTION 6: LABEL FILE INSPECTION
# --------------------------------------------------------------------------
def section6_label_inspection():
    print("=" * 70)
    print("SECTION 6: LABEL FILE INSPECTION")
    print("=" * 70)

    label_files = sorted(f for f in os.listdir(TRAIN_LABELS_DIR) if f.endswith(LABEL_EXT))
    if not label_files:
        print("No label files found.")
        print()
        return

    label_path = os.path.join(TRAIN_LABELS_DIR, label_files[0])
    with open(label_path, "r") as f:
        content = f.read()

    print(f"Label file: {label_path}")
    print("Contents:")
    print(content)

    print("YOLO format explanation:")
    print("  Each line represents one bounding box, formatted as:")
    print("  class_id center_x center_y width height")
    print("  - class_id  : integer id of the object class (e.g. 0 = tooth)")
    print("  - center_x  : box center x-coordinate, normalized 0-1 (fraction of image width)")
    print("  - center_y  : box center y-coordinate, normalized 0-1 (fraction of image height)")
    print("  - width     : box width, normalized 0-1 (fraction of image width)")
    print("  - height    : box height, normalized 0-1 (fraction of image height)")
    print()


# --------------------------------------------------------------------------
# SECTION 7: SUMMARY STATISTICS
# --------------------------------------------------------------------------
def section7_summary(train_pairs, val_pairs):
    print("=" * 70)
    print("SECTION 7: SUMMARY STATISTICS")
    print("=" * 70)

    print(f"Total image-mask pairs in train: {len(train_pairs)}")
    print(f"Total image-mask pairs in val  : {len(val_pairs)}")

    dim_sample = sample_pairs(train_pairs + val_pairs, DIM_SAMPLE_SIZE)
    heights, widths = [], []
    for image_path, _ in dim_sample:
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        heights.append(h)
        widths.append(w)

    avg_h = np.mean(heights) if heights else 0
    avg_w = np.mean(widths) if widths else 0
    print(f"Average image dimensions (sample of {len(dim_sample)}): "
          f"{avg_w:.1f} x {avg_h:.1f} (width x height)")

    all_good = len(train_pairs) > 0 and len(val_pairs) > 0
    print(f"Confirmation: dataset looks correct = {all_good}")
    print()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    section1_version_check()
    section2_folder_structure()

    train_pairs = find_matching_pairs(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    val_pairs = find_matching_pairs(VAL_IMAGES_DIR, VAL_MASKS_DIR)

    display_sample = sample_pairs(train_pairs, NUM_DISPLAY_SAMPLES)
    overlay_sample = sample_pairs(train_pairs, NUM_OVERLAY_SAMPLES)

    section3_display_images(display_sample)
    section4_display_masks(display_sample)
    section5_overlay_check(overlay_sample)
    section6_label_inspection()
    section7_summary(train_pairs, val_pairs)
