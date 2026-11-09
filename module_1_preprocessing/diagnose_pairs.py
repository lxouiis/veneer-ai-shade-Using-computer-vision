# Scratch diagnostic: why do images/masks not match 1:1 as expected?
# Not part of the main pipeline - paste into a Kaggle cell, run once, then discard.

import os
from collections import Counter

DATASET_ROOT = "/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4"


def diagnose_split(split_name):
    images_dir = os.path.join(DATASET_ROOT, split_name, "images")
    masks_dir = os.path.join(DATASET_ROOT, split_name, "masks")

    images = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg")))
    masks = sorted(os.listdir(masks_dir))

    image_bases = {os.path.splitext(f)[0] for f in images}
    mask_bases = {f[:-len("_mask.png")] for f in masks if f.endswith("_mask.png")}

    unmatched_images = sorted(image_bases - mask_bases)
    unmatched_masks = sorted(mask_bases - image_bases)

    print("=" * 70)
    print(f"SPLIT: {split_name}")
    print("=" * 70)
    print(f"Total image files: {len(images)}")
    print(f"Total mask files: {len(masks)}")
    print(f"Unique image base names: {len(image_bases)}")
    print(f"Unique mask base names: {len(mask_bases)}")
    print(f"Images with no matching mask: {len(unmatched_images)}")
    print(f"Masks with no matching image: {len(unmatched_masks)}")

    print("\nFirst 10 unmatched image base names (and their actual filenames):")
    for base in unmatched_images[:10]:
        actual = [f for f in images if f.startswith(base)]
        print(f"  {base}  ->  {actual}")

    print("\nFirst 10 unmatched mask base names (and their actual filenames):")
    for base in unmatched_masks[:10]:
        actual = [f for f in masks if f.startswith(base)]
        print(f"  {base}  ->  {actual}")

    # Check for duplicate base names across .jpg/.jpeg (same tooth photographed
    # twice under both extensions would collapse to one base name in the set
    # above but show up as 2 files here).
    base_counts = Counter(os.path.splitext(f)[0] for f in images)
    dupes = {k: v for k, v in base_counts.items() if v > 1}
    print(f"\nImage base names appearing more than once (e.g. both .jpg and .jpeg): {len(dupes)}")
    for base, count in list(dupes.items())[:10]:
        print(f"  {base}: {count} files")
    print()


diagnose_split("train")
diagnose_split("val")
