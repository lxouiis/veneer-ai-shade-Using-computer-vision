import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.losses import DiceLoss
from segment_anything import sam_model_registry

# --------------------------------------------------------------------------
# CONSTANTS / HYPERPARAMETERS
# --------------------------------------------------------------------------
CHECKPOINT_PATH = "work_dir/SAM/sam_vit_b_01ec64.pth"  # original SAM ViT-B weights to fine-tune from
OUTPUT_DIR = "/kaggle/working/checkpoints"
BEST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "medsam_tooth_best.pth")
DATA_DIR = "/kaggle/working/processed"  # output of module_1_preprocessing

# If running modules 0-5 as one merged Kaggle notebook (train + infer in a
# single session), re-running "Save & Run All" would otherwise retrain from
# scratch every time and burn GPU hours. Set to True to force a fresh train
# even if a checkpoint already exists.
FORCE_RETRAIN = False

# Set to True to run additional training starting from BEST_CHECKPOINT_PATH
# (if it exists) instead of the original base SAM weights - for a follow-up
# improvement pass (e.g. with augmentation, or unfreezing the encoder) that
# builds on a previous run rather than throwing its progress away. Has no
# effect if FORCE_RETRAIN is also True (that always starts from the original
# base SAM weights) or if BEST_CHECKPOINT_PATH doesn't exist yet.
RESUME_FROM_BEST = True

# Data augmentation applied only to the training set (never validation),
# since our earlier run had none at all - a likely reason it plateaued well
# below published benchmark numbers. Both are photometric/geometric only;
# the mask is flipped along with the image to stay consistent, never altered
# otherwise, since it's ground truth.
AUGMENT_TRAIN = True
FLIP_PROB = 0.5
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)

# Safety net for unattended "Save & Run All" runs: stop training on our own
# terms before Kaggle's session limit kills the whole run and (if it never
# reaches a normal finish) potentially discards the output entirely. Default
# 6h leaves margin under Kaggle's ~9h single-session cap for the earlier
# cells (installs, preprocessing) and checkpoint saving overhead. Whatever
# the best checkpoint was at cutoff is already saved to disk either way.
MAX_TRAINING_SECONDS = 6 * 60 * 60

# Lowered from 100 for this trial: with the encoder unfrozen, per-epoch time
# is unknown and could be meaningfully slower - better to see real timing on
# a shorter run first than commit blindly to a long unattended one. Raise
# this once you've seen epoch 1-2's actual s/epoch and know it's safe.
NUM_EPOCHS = 20
# Must stay 1: SAM's mask_decoder.predict_masks() internally does
# torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0), which
# assumes "1 image, N prompts" (e.g. N clicks on the same photo) - not "N
# different images, 1 prompt each" like a normal training batch. With
# BATCH_SIZE > 1 this silently multiplies the image batch dimension again on
# top of the real batch, causing a shape mismatch inside the decoder. Fixing
# this properly would mean looping per-image inside medsam_forward (or
# rewriting predict_masks); simplest and standard for MedSAM fine-tuning is
# to just process one image per forward pass.
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
# Only used when FREEZE_IMAGE_ENCODER = False. Deliberately much smaller than
# LEARNING_RATE: the encoder already has good pretrained features from SAM's
# original training - a full-size LR here risks destroying them (catastrophic
# forgetting) rather than gently adapting them to tooth photos. 10x smaller
# than the decoder's LR is a common starting point for this kind of
# differential fine-tuning.
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
BBOX_SHIFT = 20          # random jitter (pixels) added to the GT bbox prompt, for robustness
VAL_EVERY_N_EPOCHS = 5
IMG_SIZE = 1024           # must match module_1_preprocessing output size
NUM_WORKERS = 2
RANDOM_SEED = 42

# A T4 has 16GB VRAM. With BATCH_SIZE forced to 1 (see above), a trainable
# ViT-B image encoder at 1024x1024 may now actually fit with AMP - worth
# trying as a follow-up experiment (see ENCODER_LEARNING_RATE), but real OOM
# risk remains. If this OOMs, set it back to True and fall back to the
# already-good frozen-encoder checkpoint - RESUME_FROM_BEST's re-validation
# step means a failed/worse attempt here can't overwrite it.
FREEZE_IMAGE_ENCODER = False
USE_AMP = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# BBOX PROMPT FROM GROUND-TRUTH MASK
# --------------------------------------------------------------------------
def get_bbox_from_mask(mask):
    # Find every pixel belonging to the tooth so we can bound them.
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    h, w = mask.shape

    # Jitter the box randomly by up to BBOX_SHIFT pixels on each side. This
    # simulates an imprecise user/detector-drawn box so the model doesn't
    # overfit to perfectly tight boxes.
    x_min = max(0, x_min - random.randint(0, BBOX_SHIFT))
    x_max = min(w, x_max + random.randint(0, BBOX_SHIFT))
    y_min = max(0, y_min - random.randint(0, BBOX_SHIFT))
    y_max = min(h, y_max + random.randint(0, BBOX_SHIFT))

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


# --------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------
def load_medsam_model(checkpoint_path):
    model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)

    # Freeze the prompt encoder: we always prompt with a plain bounding box,
    # so there's nothing task-specific for it to learn, and freezing it saves
    # memory/compute.
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False

    # Freeze the image encoder too (see FREEZE_IMAGE_ENCODER above): this is
    # what actually keeps training inside a T4's 16GB, since backprop through
    # a trainable ViT-B encoder at 1024x1024 needs far more memory than that.
    # Only the mask decoder is fine-tuned on tooth photos.
    if FREEZE_IMAGE_ENCODER:
        for param in model.image_encoder.parameters():
            param.requires_grad = False

    return model.to(DEVICE)


# --------------------------------------------------------------------------
# DATASET
# --------------------------------------------------------------------------
def augment_pair(image, mask):
    # Random horizontal flip - applied to both image and mask together so the
    # mask stays geometrically consistent with the (now-flipped) image.
    if random.random() < FLIP_PROB:
        image = np.ascontiguousarray(image[:, ::-1, :])
        mask = np.ascontiguousarray(mask[:, ::-1])

    # Random brightness/contrast jitter - image only, since this is purely
    # photometric and the mask's ground-truth tooth region doesn't change
    # just because the photo got brighter or lower-contrast.
    brightness = random.uniform(*BRIGHTNESS_RANGE)
    contrast = random.uniform(*CONTRAST_RANGE)
    image = image.astype(np.float32)
    mean = image.mean()
    image = (image - mean) * contrast + mean * brightness
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image, mask


class ToothDataset(Dataset):
    def __init__(self, data_dir, split, augment=False):
        self.images_dir = os.path.join(data_dir, split, "images")
        self.masks_dir = os.path.join(data_dir, split, "masks")
        self.filenames = sorted(
            f for f in os.listdir(self.images_dir) if f.endswith(".npy")
        )
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = np.load(os.path.join(self.images_dir, filename))  # (H, W, 3) uint8, 0-255
        mask = np.load(os.path.join(self.masks_dir, filename))    # (H, W) uint8, 0/1

        if self.augment:
            image, mask = augment_pair(image, mask)

        # module_1 stores images as uint8 to save disk; normalize to 0-1 here.
        image = image.astype(np.float32) / 255.0

        bbox = get_bbox_from_mask(mask)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # -> (3, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()        # -> (1, H, W)
        bbox_tensor = torch.from_numpy(bbox).float()

        return image_tensor, mask_tensor, bbox_tensor


# --------------------------------------------------------------------------
# MEDSAM FORWARD PASS (image encoder -> frozen prompt encoder -> mask decoder)
# --------------------------------------------------------------------------
def medsam_forward(model, image, bbox):
    # Running the (frozen) image encoder under no_grad is what actually saves
    # the memory: it stops PyTorch retaining encoder activations for backprop.
    if FREEZE_IMAGE_ENCODER:
        with torch.no_grad():
            image_embedding = model.image_encoder(image)  # (B, 256, 64, 64)
    else:
        image_embedding = model.image_encoder(image)

    box_torch = bbox
    if box_torch.dim() == 2:
        box_torch = box_torch.unsqueeze(1)  # (B, 1, 4), as expected by the prompt encoder

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None, boxes=box_torch, masks=None,
        )

    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    # Upsample from the decoder's low-res output back to full image resolution.
    return F.interpolate(low_res_masks, size=image.shape[-2:], mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------
def dice_score(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    return ((2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)).item()


def iou_score(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()


# --------------------------------------------------------------------------
# TRAIN / VALIDATE LOOPS
# --------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0.0

    for images, masks, bboxes in dataloader:
        images, masks, bboxes = images.to(device), masks.to(device), bboxes.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=USE_AMP):
            pred_logits = medsam_forward(model, images, bboxes)
            loss = loss_fn(pred_logits, masks)

        # GradScaler keeps small gradients from underflowing to zero in fp16.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    dsc_scores, iou_scores = [], []

    with torch.no_grad():
        for images, masks, bboxes in dataloader:
            images, masks, bboxes = images.to(device), masks.to(device), bboxes.to(device)

            with torch.amp.autocast(device.type, enabled=USE_AMP):
                pred_logits = medsam_forward(model, images, bboxes)
            pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()

            dsc_scores.append(dice_score(pred_binary, masks))
            iou_scores.append(iou_score(pred_binary, masks))

    return float(np.mean(dsc_scores)), float(np.mean(iou_scores))


# --------------------------------------------------------------------------
# COMBINED LOSS: DICE + BCE
# --------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, pred_logits, target):
        return self.dice_loss(pred_logits, target) + self.bce_loss(pred_logits, target)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    model = load_medsam_model(CHECKPOINT_PATH)

    # Lets modules 0-5 be run as one merged Kaggle notebook: re-running all
    # cells (e.g. via "Save & Run All") loads the existing best checkpoint
    # instead of paying for a full retrain again. See FORCE_RETRAIN above.
    # RESUME_FROM_BEST opts back into training even when a checkpoint exists,
    # continuing from it instead of skipping or restarting from scratch.
    already_trained = os.path.exists(BEST_CHECKPOINT_PATH) and not FORCE_RETRAIN and not RESUME_FROM_BEST

    if already_trained:
        print(f"Found existing checkpoint at {BEST_CHECKPOINT_PATH} - loading it and skipping training.")
        print("Set FORCE_RETRAIN = True (fresh) or RESUME_FROM_BEST = True (continue) to train more.")
        model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE))
    else:
        train_dataset = ToothDataset(DATA_DIR, "train", augment=AUGMENT_TRAIN)
        val_dataset = ToothDataset(DATA_DIR, "val", augment=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, augment={AUGMENT_TRAIN}")

        if FREEZE_IMAGE_ENCODER:
            # Image encoder is frozen, so only the mask decoder has
            # trainable parameters.
            optimizer = torch.optim.AdamW(
                model.mask_decoder.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            # Differential learning rates: the decoder keeps training at the
            # normal rate, but the newly-unfrozen encoder gets a much smaller
            # one (see ENCODER_LEARNING_RATE above) so it adapts gently
            # instead of overwriting its pretrained features.
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.mask_decoder.parameters(), "lr": LEARNING_RATE},
                    {"params": model.image_encoder.parameters(), "lr": ENCODER_LEARNING_RATE},
                ],
                weight_decay=WEIGHT_DECAY,
            )
        loss_fn = DiceBCELoss()
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

        best_dsc = 0.0
        if RESUME_FROM_BEST and os.path.exists(BEST_CHECKPOINT_PATH) and not FORCE_RETRAIN:
            print(f"Resuming from {BEST_CHECKPOINT_PATH} for {NUM_EPOCHS} more epochs.")
            model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE))
            # Re-validate the resumed checkpoint first so best_dsc reflects
            # what it can actually already do - otherwise an early epoch that
            # happens to score worse (e.g. while adapting to augmentation)
            # could overwrite a better checkpoint with a worse one.
            best_dsc, best_iou = validate(model, val_loader, DEVICE)
            print(f"Resumed checkpoint's current val DSC: {best_dsc:.4f}, IoU: {best_iou:.4f}")
        elif RESUME_FROM_BEST:
            # Loud on purpose: silently falling through to a full fresh
            # retrain from base SAM weights (because BEST_CHECKPOINT_PATH
            # isn't in THIS session's /kaggle/working - e.g. it only exists
            # in a previous committed run's Output) would otherwise look
            # identical to a real resume in the logs until you noticed the
            # DSC starting from scratch instead of picking up where you left
            # off.
            print(f"WARNING: RESUME_FROM_BEST=True but no checkpoint found at "
                  f"{BEST_CHECKPOINT_PATH} in this session - training from the "
                  f"original base SAM weights instead. If you meant to resume, "
                  f"attach the previous run's Output and copy the checkpoint "
                  f"into place before running this cell.")

        epoch_times = []
        training_start = time.time()
        for epoch in range(1, NUM_EPOCHS + 1):
            elapsed = time.time() - training_start
            if elapsed > MAX_TRAINING_SECONDS:
                print(f"Hit MAX_TRAINING_SECONDS ({MAX_TRAINING_SECONDS}s) after "
                      f"{epoch - 1} epochs - stopping early. Best checkpoint already "
                      f"saved to {BEST_CHECKPOINT_PATH}.")
                break

            epoch_start = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler)
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining = avg_epoch_time * (NUM_EPOCHS - epoch)
            print(f"Epoch {epoch}/{NUM_EPOCHS} - train loss: {train_loss:.4f} "
                  f"- {epoch_time:.1f}s/epoch (avg {avg_epoch_time:.1f}s, "
                  f"~{remaining / 60:.1f} min remaining)")

            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
                val_dsc, val_iou = validate(model, val_loader, DEVICE)
                print(f"  -> val DSC: {val_dsc:.4f}, val IoU: {val_iou:.4f}")

                if val_dsc > best_dsc:
                    best_dsc = val_dsc
                    torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
                    print(f"  -> new best DSC, saved checkpoint to {BEST_CHECKPOINT_PATH}")

        print(f"Training complete. Best val DSC achieved: {best_dsc:.4f}")
