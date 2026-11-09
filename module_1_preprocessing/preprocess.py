import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_SIZE = 1024
IMAGE_EXTENSIONS = (".jpg", ".jpeg")  # dataset has both extensions mixed in
MASK_SUFFIX = "_mask.png"
MASK_BINARY_THRESHOLD = 127

DATASET_ROOT = "/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4"
OUTPUT_ROOT = "/kaggle/working/processed"


def get_mask_filename(image_filename):
    base_name, _ = os.path.splitext(image_filename)
    return base_name + MASK_SUFFIX


def find_matching_pairs(images_dir, masks_dir):
    pairs = []
    for image_filename in sorted(os.listdir(images_dir)):
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        mask_path = os.path.join(masks_dir, get_mask_filename(image_filename))
        if os.path.exists(mask_path):
            pairs.append((os.path.join(images_dir, image_filename), mask_path))
    return pairs


def process_image(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # cv2.INTER_CUBIC is the same bicubic interpolation as skimage's order=3,
    # but OpenCV's native implementation is dramatically faster - no separate
    # Python-level anti-aliasing pass, which was the main preprocessing
    # bottleneck on large source photos.
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    # Stored as uint8 (0-255), not normalized float32, to keep disk usage down
    # (~4x smaller); normalization happens on load in the training Dataset instead.
    return image_resized.astype(np.uint8)


def process_mask(mask_path):
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # INTER_NEAREST is cv2's nearest-neighbor mode - same reasoning as before:
    # masks are label maps, so we must not interpolate between class values.
    mask_resized = cv2.resize(mask_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask_resized > MASK_BINARY_THRESHOLD).astype(np.uint8)


def process_split(split_name):
    images_dir = os.path.join(DATASET_ROOT, split_name, "images")
    masks_dir = os.path.join(DATASET_ROOT, split_name, "masks")
    out_images_dir = os.path.join(OUTPUT_ROOT, split_name, "images")
    out_masks_dir = os.path.join(OUTPUT_ROOT, split_name, "masks")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    pairs = find_matching_pairs(images_dir, masks_dir)
    print(f"[{split_name}] found {len(pairs)} matched image/mask pairs")

    first_saved_paths = None
    for image_path, mask_path in tqdm(pairs, desc=f"[{split_name}] preprocessing"):
        image_array = process_image(image_path)
        mask_array = process_mask(mask_path)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_image_path = os.path.join(out_images_dir, base_name + ".npy")
        out_mask_path = os.path.join(out_masks_dir, base_name + ".npy")

        np.save(out_image_path, image_array)
        np.save(out_mask_path, mask_array)

        if first_saved_paths is None:
            first_saved_paths = (out_image_path, out_mask_path)

    return first_saved_paths


def visualize_sample(image_npy_path, mask_npy_path):
    image_array = np.load(image_npy_path)
    mask_array = np.load(mask_npy_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_array)
    axes[0].set_title("Preprocessed Image")
    axes[0].axis("off")

    axes[1].imshow(mask_array, cmap="gray")
    axes[1].set_title("Binary Mask")
    axes[1].axis("off")

    axes[2].imshow(image_array)
    axes[2].imshow(mask_array, cmap="Reds", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_sample = process_split("train")
    val_sample = process_split("val")

    sample_to_show = train_sample if train_sample is not None else val_sample
    if sample_to_show is not None:
        visualize_sample(*sample_to_show)
    else:
        print("No processed samples found - check that DATASET_ROOT is correct.")
