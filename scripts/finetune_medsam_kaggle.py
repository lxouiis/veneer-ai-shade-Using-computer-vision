#!/usr/bin/env python3
"""
scripts/finetune_medsam_kaggle.py

Kaggle-Ready MedSAM Fine-Tuning Script
--------------------------------------
Answers: "Which checkpoint do I fine-tune from?"
 -> Answer: You fine-tune MedSAM using `medsam_tooth_best.pth` (from Notebook_A_meowmeow / DentalMedSAM)!
    YOLO is NOT for fine-tuning MedSAM — YOLO is purely a bounding box detector.

Features:
 1. Automatically searches for `medsam_tooth_best.pth` across Kaggle datasets & local dirs.
 2. Resumes training from your existing fine-tuned weights for maximum accuracy.
 3. Supports differential learning rate (unfreezes image encoder safely).
 4. Automatic Mixed Precision (AMP) + native SoftDice & BCE loss.
"""

import os
import sys
import glob
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry

# --------------------------------------------------------------------------
# PATH RESOLUTION & CHECKPOINT SEARCH
# --------------------------------------------------------------------------
def find_checkpoint():
    """
    Searches for the existing fine-tuned MedSAM checkpoint in Kaggle input datasets
    or local working directories.
    """
    candidates = [
        "/kaggle/input/notebook-a-meowmeow/checkpoints/medsam_tooth_best.pth",
        "/kaggle/input/datasets/hemantdarur/notebook-a-meowmeow/checkpoints/medsam_tooth_best.pth",
        "/kaggle/input/dentalmedsam/checkpoints/medsam_tooth_best.pth",
        "/kaggle/working/checkpoints/medsam_tooth_best.pth",
        "./checkpoints/medsam_tooth_best.pth",
        "../checkpoints/medsam_tooth_best.pth"
    ]
    # Also search any wildcard under /kaggle/input/
    wildcard_matches = glob.glob("/kaggle/input/**/medsam_tooth_best.pth", recursive=True)
    candidates.extend(wildcard_matches)

    for p in candidates:
        if os.path.exists(p):
            print(f"[Checkpoint Found] Using pre-trained tooth checkpoint: {p}")
            return p

    print("[Info] No existing medsam_tooth_best.pth found in input datasets.")
    print("      Will initialize from base SAM ViT-B weights.")
    return None

def find_data_dir():
    """Locates processed dataset directory (.npy images and masks)."""
    candidates = [
        "/kaggle/working/processed",
        "/kaggle/input/notebook-a-meowmeow/processed",
        "/kaggle/input/dentalmedsam/processed",
        "./data/processed",
        "../data/processed"
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train", "images")):
            print(f"[Dataset Found] Using processed data directory: {p}")
            return p
    return "./data/processed"

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
PRETRAINED_CHECKPOINT = find_checkpoint()
BASE_SAM_WEIGHTS = "work_dir/SAM/sam_vit_b_01ec64.pth"
OUTPUT_DIR = "/kaggle/working/checkpoints" if os.path.exists("/kaggle/working") else "./checkpoints"
BEST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "medsam_tooth_best.pth")
DATA_DIR = find_data_dir()

NUM_EPOCHS = 20
BATCH_SIZE = 1  # SAM predict_masks requires batch_size=1 during training
LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
BBOX_SHIFT = 20
VAL_EVERY_N_EPOCHS = 5
FREEZE_IMAGE_ENCODER = False  # Set to False to adapt Vision Transformer encoder
USE_AMP = True
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# LOSS FUNCTIONS
# --------------------------------------------------------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets)

# --------------------------------------------------------------------------
# DATASET & AUGMENTATIONS
# --------------------------------------------------------------------------
def get_bbox_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return np.array([0, 0, 1024, 1024], dtype=np.float32)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    h, w = mask.shape

    x_min = max(0, x_min - random.randint(0, BBOX_SHIFT))
    x_max = min(w, x_max + random.randint(0, BBOX_SHIFT))
    y_min = max(0, y_min - random.randint(0, BBOX_SHIFT))
    y_max = min(h, y_max + random.randint(0, BBOX_SHIFT))
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

class ToothDataset(Dataset):
    def __init__(self, data_dir, split, augment=False):
        self.images_dir = os.path.join(data_dir, split, "images")
        self.masks_dir = os.path.join(data_dir, split, "masks")
        if os.path.exists(self.images_dir):
            self.filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".npy")])
        else:
            self.filenames = []
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = np.load(os.path.join(self.images_dir, filename))
        mask = np.load(os.path.join(self.masks_dir, filename))

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1, :])
                mask = np.ascontiguousarray(mask[:, ::-1])

        image = image.astype(np.float32) / 255.0
        bbox = get_bbox_from_mask(mask)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        bbox_tensor = torch.from_numpy(bbox).float()

        return image_tensor, mask_tensor, bbox_tensor

# --------------------------------------------------------------------------
# MEDSAM FORWARD PASS
# --------------------------------------------------------------------------
def medsam_forward(model, image, bbox):
    if FREEZE_IMAGE_ENCODER:
        with torch.no_grad():
            image_embedding = model.image_encoder(image)
    else:
        image_embedding = model.image_encoder(image)

    box_torch = bbox if bbox.dim() == 3 else bbox.unsqueeze(1)

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )

    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    return F.interpolate(low_res_masks, size=image.shape[-2:], mode="bilinear", align_corners=False)

# --------------------------------------------------------------------------
# METRICS & TRAINING LOOP
# --------------------------------------------------------------------------
def compute_dice(pred_binary, target):
    intersection = (pred_binary * target).sum()
    return ((2.0 * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)).item()

def validate(model, dataloader):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, masks, bboxes in dataloader:
            images, masks, bboxes = images.to(DEVICE), masks.to(DEVICE), bboxes.to(DEVICE)
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                pred_logits = medsam_forward(model, images, bboxes)
            pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
            scores.append(compute_dice(pred_binary, masks))
    return float(np.mean(scores)) if scores else 0.0

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Running on Device: {DEVICE}")

    # 1. Load Model Architecture
    model = sam_model_registry["vit_b"](checkpoint=None)

    # Freeze prompt encoder
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False

    # 2. Load Weights (Prioritize existing tooth checkpoint -> fallback to base SAM)
    weights_to_load = PRETRAINED_CHECKPOINT if PRETRAINED_CHECKPOINT else BASE_SAM_WEIGHTS
    if os.path.exists(weights_to_load):
        print(f"Loading checkpoint weights from: {weights_to_load}")
        state_dict = torch.load(weights_to_load, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint {weights_to_load} not found on disk. Initializing base SAM model.")

    if FREEZE_IMAGE_ENCODER:
        for param in model.image_encoder.parameters():
            param.requires_grad = False

    model.to(DEVICE)

    # 3. Data Loaders
    train_dataset = ToothDataset(DATA_DIR, "train", augment=True)
    val_dataset = ToothDataset(DATA_DIR, "val", augment=False)

    if len(train_dataset) == 0:
        print(f"Error: No training files found in {DATA_DIR}/train/images.")
        print("Please check your DATA_DIR path or run dataset preprocessing first.")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Dataset Loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # 4. Optimizer
    if FREEZE_IMAGE_ENCODER:
        optimizer = torch.optim.AdamW(model.mask_decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.mask_decoder.parameters(), "lr": LEARNING_RATE},
            {"params": model.image_encoder.parameters(), "lr": ENCODER_LEARNING_RATE}
        ], weight_decay=WEIGHT_DECAY)

    criterion = CombinedLoss()
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

    best_dsc = validate(model, val_loader) if len(val_dataset) > 0 else 0.0
    print(f"Initial Validation DSC Baseline: {best_dsc:.4f}")

    # 5. Training Loop
    print("\n--- Starting MedSAM Fine-Tuning ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for images, masks, bboxes in train_loader:
            images, masks, bboxes = images.to(DEVICE), masks.to(DEVICE), bboxes.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                logits = medsam_forward(model, images, bboxes)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            val_dsc = validate(model, val_loader)
            print(f"  -> Validation DSC: {val_dsc:.4f}")
            if val_dsc > best_dsc:
                best_dsc = val_dsc
                torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
                print(f"  -> New Best Checkpoint Saved to {BEST_CHECKPOINT_PATH} (DSC: {val_dsc:.4f})")

    print(f"\nFine-Tuning Complete! Best Validation DSC: {best_dsc:.4f}")

if __name__ == "__main__":
    main()
