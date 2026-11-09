import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# Evaluates the checkpoint resumed in docs/finetune.ipynb (NCT + SmileImages,
# image encoder unfrozen) the way the real pipeline actually runs inference -
# with a YOLO-predicted box, not the GT-mask-derived box finetune.ipynb's own
# validate() uses internally. That GT-derived box is only a fair comparison
# against previous training runs, not a measure of real-world accuracy, since
# a new patient photo never comes with a ground-truth mask to derive a box
# from (see module_3_medsam_inference/notes.md).
def find_checkpoint():
    candidates = [
        "/kaggle/working/checkpoints/medsam_tooth_best.pth",  # same-session finetune.ipynb output
        "/kaggle/input/notebook-a-meowmeow/checkpoints/medsam_tooth_best.pth",
        "/kaggle/input/datasets/hemantdarur/notebook-a-meowmeow/checkpoints/medsam_tooth_best.pth",
    ]
    candidates.extend(glob.glob("/kaggle/input/**/medsam_tooth_best.pth", recursive=True))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No medsam_tooth_best.pth found. Run this in the same session as "
        "docs/finetune.ipynb (so /kaggle/working/checkpoints has the new "
        "weights), or publish that checkpoint as a Kaggle Dataset first."
    )


def find_data_dir():
    # /kaggle/working/processed is where finetune.ipynb's cell 3 merges NCT
    # + SmileImages .npy files into one train/val split - same-session only.
    candidates = [
        "/kaggle/working/processed",
        "/kaggle/input/notebook-a-meowmeow/processed",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "val", "images")):
            return p
    raise FileNotFoundError(
        "No combined val/images directory found. Run this in the same "
        "session as docs/finetune.ipynb right after its merge cell, or "
        "publish /kaggle/working/processed as a Kaggle Dataset first."
    )


CHECKPOINT_PATH = find_checkpoint()
DATA_DIR = find_data_dir()

# Same trained detector used throughout module_3/3c - this eval is scoped to
# "does the new checkpoint work well under the real YOLO-box pipeline",
# nothing about localization changes here.
YOLO_CHECKPOINT_PATH = "/kaggle/input/datasets/hemantdarur/yolomodel-inference/yolo_checkpoints/tooth_yolo/weights/best.pt"
YOLO_CONF_THRESHOLD = 0.25
YOLO_BBOX_MARGIN = 20  # matches module_3's margin - MedSAM trained on boxes jittered by up to this much

IMG_SIZE = 1024  # processed .npy images are already 1024x1024 uint8, same as training
MASK_BINARY_THRESHOLD = 0.5

OUTPUT_DIR = "/kaggle/working/outputs_eval_combined"
NUM_WORST_TO_SHOW = 15  # the failing tail - what's actually dragging the mean below the median
NUM_BEST_TO_SHOW = 10   # a sanity check that "best" isn't trivially easy/degenerate cases

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# DATASET
# --------------------------------------------------------------------------
class ToothValDataset(Dataset):
    """Returns the raw uint8 image (for YOLO) alongside the GT mask - unlike
    finetune.ipynb's ToothDataset, this does NOT compute a box itself."""

    def __init__(self, data_dir):
        self.images_dir = os.path.join(data_dir, "val", "images")
        self.masks_dir = os.path.join(data_dir, "val", "masks")
        self.filenames = sorted(
            f for f in os.listdir(self.images_dir) if f.endswith(".npy")
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = np.load(os.path.join(self.images_dir, filename))  # uint8 HxWx3
        mask = np.load(os.path.join(self.masks_dir, filename))    # 0/1 HxW
        return filename, image, mask


# --------------------------------------------------------------------------
# YOLO BOX PROMPT (same recipe as module_3/3c's get_bbox_prompt_yolo)
# --------------------------------------------------------------------------
def get_yolo_bbox(yolo_model, image_uint8, margin=YOLO_BBOX_MARGIN):
    results = yolo_model.predict(image_uint8, conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if boxes.shape[0] == 0:
        return None

    x_min, y_min = boxes[:, 0].min(), boxes[:, 1].min()
    x_max, y_max = boxes[:, 2].max(), boxes[:, 3].max()

    h, w = image_uint8.shape[:2]
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


# --------------------------------------------------------------------------
# MEDSAM FORWARD (identical to finetune.ipynb's medsam_forward, eval mode)
# --------------------------------------------------------------------------
def medsam_forward(model, image, bbox):
    with torch.no_grad():
        image_embedding = model.image_encoder(image)
        box_torch = bbox if bbox.dim() == 3 else bbox.unsqueeze(1)
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
# METRICS
# --------------------------------------------------------------------------
def compute_dice_iou(pred_binary, target):
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    dice = ((2.0 * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)).item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).item()
    return dice, iou


# --------------------------------------------------------------------------
# WORST/BEST VISUALIZATION
# --------------------------------------------------------------------------
def run_single(model, image, bbox):
    """Re-runs one sample through the model - used for the worst/best
    re-pass so the main scoring loop doesn't have to hold every prediction
    in memory for all 238 images just to maybe visualize a handful."""
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    bbox_tensor = torch.from_numpy(bbox).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=torch.cuda.is_available()):
        logits = medsam_forward(model, image_tensor, bbox_tensor)
    pred_binary = (torch.sigmoid(logits) > MASK_BINARY_THRESHOLD).float()
    return pred_binary.squeeze().cpu().numpy()


def visualize_case(image, gt_mask, pred_mask, dice, iou, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(f"Predicted (DSC={dice:.3f}, IoU={iou:.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Combined val set: {DATA_DIR}/val")

    model = sam_model_registry["vit_b"](checkpoint=None)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"mask_decoder class: {type(model.mask_decoder).__name__}  "
          f"(expect 'MaskDecoder', not 'MaskDecoderHQ' - see module notes)")

    yolo_model = YOLO(YOLO_CHECKPOINT_PATH)
    val_dataset = ToothValDataset(DATA_DIR)
    print(f"Combined val samples: {len(val_dataset)}")

    # Per-image results, not just running totals - needed to rank
    # worst/best afterward. bbox is cached per filename so the worst/best
    # re-pass below doesn't have to re-run YOLO.
    results = []  # list of dicts: filename, dice, iou, bbox
    skipped = 0

    with torch.no_grad():
        for filename, image, mask in val_dataset:
            bbox = get_yolo_bbox(yolo_model, image)
            if bbox is None:
                skipped += 1
                continue

            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            bbox_tensor = torch.from_numpy(bbox).unsqueeze(0).to(DEVICE)

            with torch.amp.autocast(DEVICE.type, enabled=torch.cuda.is_available()):
                logits = medsam_forward(model, image_tensor, bbox_tensor)
            pred_binary = (torch.sigmoid(logits) > MASK_BINARY_THRESHOLD).float()

            dice, iou = compute_dice_iou(pred_binary, mask_tensor)
            results.append({"filename": filename, "dice": dice, "iou": iou, "bbox": bbox})

    dice_scores = [r["dice"] for r in results]
    iou_scores = [r["iou"] for r in results]

    print(f"\nEvaluated {len(results)} / {len(val_dataset)} images "
          f"({skipped} skipped - YOLO found no detection, consistent with "
          f"module_3c's no-fallback convention)")
    print(f"Mean DSC: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"Median DSC: {np.median(dice_scores):.4f}  (less skewed by a few bad outliers than the mean)")

    # --- Worst / best breakdown ---
    ranked = sorted(results, key=lambda r: r["dice"])
    worst = ranked[:NUM_WORST_TO_SHOW]
    best = ranked[-NUM_BEST_TO_SHOW:][::-1]

    print(f"\n--- Worst {len(worst)} (lowest DSC - likely what's dragging the mean down) ---")
    for r in worst:
        print(f"  {r['filename']}: DSC={r['dice']:.4f}  IoU={r['iou']:.4f}")

    print(f"\n--- Best {len(best)} (highest DSC) ---")
    for r in best:
        print(f"  {r['filename']}: DSC={r['dice']:.4f}  IoU={r['iou']:.4f}")

    # Re-run just these ~25 images to save visual comparisons - cheap
    # relative to the full 238-image pass above, and loads only these
    # files from disk rather than holding all 238 in memory at once.
    for rank_label, group in [("worst", worst), ("best", best)]:
        for i, r in enumerate(group):
            image = np.load(os.path.join(val_dataset.images_dir, r["filename"]))
            mask = np.load(os.path.join(val_dataset.masks_dir, r["filename"]))
            pred_mask = run_single(model, image, r["bbox"])
            base_name = os.path.splitext(r["filename"])[0]
            save_path = os.path.join(OUTPUT_DIR, f"{rank_label}_{i+1:02d}_{base_name}.png")
            visualize_case(image, mask, pred_mask, r["dice"], r["iou"], save_path)

    print(f"\nSaved {len(worst) + len(best)} comparison visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
