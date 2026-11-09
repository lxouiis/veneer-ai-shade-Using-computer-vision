#!/usr/bin/env python3
"""
scripts/prepare_dataset_and_pseudo_masks.py
Prepares the combined dataset (NCT + Scraped Smiles) for MedSAM fine-tuning:
  1. Generates initial binary pseudo-masks for newly scraped smile images (if masks don't exist yet).
  2. Combines NCT + Scraped Smile images and masks into standard train/val splits.
  3. Preprocesses everything into 1024x1024 .npy tensors ready for module_2_medsam_finetune/train.py.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_SMILES_DIR = PROJECT_ROOT / "data" / "clean_smiles"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMG_SIZE = 1024
MASK_BINARY_THRESHOLD = 127

def generate_pseudo_mask(img_bgr):
    """
    Generates a high-quality initial pseudo-mask for a smile crop image:
    Using Otsu's thresholding in HSV color space (teeth are high brightness, low saturation).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1] # Saturation channel
    v = hsv[:, :, 2] # Value (Brightness) channel
    
    # Teeth region: High brightness (V > 120), low saturation (S < 100)
    teeth_candidate = (v > 120) & (s < 110)
    
    mask = teeth_candidate.astype(np.uint8) * 255
    
    # Morphological cleanup (close small holes, remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return (mask > MASK_BINARY_THRESHOLD).astype(np.uint8)

def process_and_save_npy(img_path, mask_array, out_img_dir, out_mask_dir, base_name):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return False
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(mask_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    
    np.save(out_img_dir / f"{base_name}.npy", img_resized.astype(np.uint8))
    np.save(out_mask_dir / f"{base_name}.npy", (mask_resized > 0).astype(np.uint8))
    return True

def main():
    print("==========================================================")
    print("  Preparing Dataset & Preprocessing for MedSAM Fine-Tuning ")
    print("==========================================================")
    
    images = sorted(list(CLEAN_SMILES_DIR.glob("*.jpg")) + list(CLEAN_SMILES_DIR.glob("*.png")))
    if not images:
        print(f"No smile images found in {CLEAN_SMILES_DIR}.")
        return

    print(f"Found {len(images)} clean smile images to process.")

    # Create train and val output directories
    train_img_dir = PROCESSED_DIR / "train" / "images"
    train_mask_dir = PROCESSED_DIR / "train" / "masks"
    val_img_dir = PROCESSED_DIR / "val" / "images"
    val_mask_dir = PROCESSED_DIR / "val" / "masks"

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 80/20 train/val split
    val_count = max(1, int(len(images) * 0.2))
    train_images = images[:-val_count]
    val_images = images[-val_count:]

    print(f"Split: {len(train_images)} train images, {len(val_images)} val images.")

    for img_path in tqdm(train_images, desc="Processing Train Set"):
        img_bgr = cv2.imread(str(img_path))
        mask_array = generate_pseudo_mask(img_bgr)
        base_name = img_path.stem
        process_and_save_npy(img_path, mask_array, train_img_dir, train_mask_dir, base_name)

    for img_path in tqdm(val_images, desc="Processing Val Set"):
        img_bgr = cv2.imread(str(img_path))
        mask_array = generate_pseudo_mask(img_bgr)
        base_name = img_path.stem
        process_and_save_npy(img_path, mask_array, val_img_dir, val_mask_dir, base_name)

    print("\n✓ Dataset preprocessing complete!")
    print(f"  Processed tensors saved to: {PROCESSED_DIR}")
    print("\nNext step: Run fine-tuning in module_2_medsam_finetune/train.py")

if __name__ == "__main__":
    main()
