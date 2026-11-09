import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from segment_anything import sam_model_registry
from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# Same as evaluate.py, except this uses the smile-finetuned YOLO detector
# (module_2b's finetune_yolo_smile.py output) instead of the original one,
# to see whether retraining YOLO on SmileImages fixes the localization
# failures seen there (e.g. mous8's box ballooning into skin/lips).
#
# MedSAM stays on the ORIGINAL frozen-encoder checkpoint (DSC 0.8838), not
# the resumed/unfrozen one from docs/finetune.ipynb - that run's own history
# (docs/EXPERIMENTS.md run 2) already showed unfreezing the encoder doesn't
# beat the frozen baseline, so this isolates "does the new YOLO box help" as
# the one variable being tested, same reasoning as module_3's YOLO-vs-generic
# box comparison.
def find_checkpoint():
    candidates = [
        "/kaggle/input/notebook-a-meowmeow/checkpoints/medsam_tooth_best.pth",
        "/kaggle/input/datasets/hemantdarur/notebook-a-meowmeow/checkpoints/medsam_tooth_best.pth",
    ]
    candidates.extend(glob.glob("/kaggle/input/**/notebook-a-meowmeow/**/medsam_tooth_best.pth", recursive=True))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No frozen-encoder medsam_tooth_best.pth found. Attach the "
        "notebook-a-meowmeow dataset to this session."
    )


def find_yolo_checkpoint():
    candidates = [
        "/kaggle/working/yolo_checkpoints_smile/tooth_yolo_smile/weights/best.pt",  # same-session finetune_yolo_smile.py output
    ]
    candidates.extend(glob.glob("/kaggle/input/**/tooth_yolo_smile/weights/best.pt", recursive=True))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No tooth_yolo_smile checkpoint found. Run this in the same session "
        "as finetune_yolo_smile.py, or publish its best.pt as a Kaggle "
        "Dataset and attach it first."
    )


def find_data_dir():
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
YOLO_CHECKPOINT_PATH = find_yolo_checkpoint()
DATA_DIR = find_data_dir()

YOLO_CONF_THRESHOLD = 0.25
YOLO_BBOX_MARGIN = 20  # matches module_3's margin - MedSAM trained on boxes jittered by up to this much

IMG_SIZE = 1024  # processed .npy images are already 1024x1024 uint8, same as training
MASK_BINARY_THRESHOLD = 0.5

OUTPUT_DIR = "/kaggle/working/outputs_eval_new_yolo"
NUM_WORST_TO_SHOW = 15
NUM_BEST_TO_SHOW = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# DATASET
# --------------------------------------------------------------------------
class ToothValDataset(Dataset):
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
        image = np.load(os.path.join(self.images_dir, filename))
        mask = np.load(os.path.join(self.masks_dir, filename))
        return filename, image, mask


# --------------------------------------------------------------------------
# YOLO BOX PROMPT
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
# MEDSAM FORWARD
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


def run_single(model, image, bbox):
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    bbox_tensor = torch.from_numpy(bbox).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=torch.cuda.is_available()):
        logits = medsam_forward(model, image_tensor, bbox_tensor)
    pred_binary = (torch.sigmoid(logits) > MASK_BINARY_THRESHOLD).float()
    return pred_binary.squeeze().cpu().numpy()


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
# VISUALIZATION
# --------------------------------------------------------------------------
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
    print(f"MedSAM checkpoint (frozen encoder): {CHECKPOINT_PATH}")
    print(f"YOLO checkpoint (smile-finetuned): {YOLO_CHECKPOINT_PATH}")
    print(f"Combined val set: {DATA_DIR}/val")

    model = sam_model_registry["vit_b"](checkpoint=None)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"mask_decoder class: {type(model.mask_decoder).__name__}  "
          f"(expect 'MaskDecoder', not 'MaskDecoderHQ')")

    yolo_model = YOLO(YOLO_CHECKPOINT_PATH)
    val_dataset = ToothValDataset(DATA_DIR)
    print(f"Combined val samples: {len(val_dataset)}")

    results = []
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
          f"({skipped} skipped - YOLO found no detection)")
    print(f"Mean DSC: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"Median DSC: {np.median(dice_scores):.4f}")
    print("\nCompare these against evaluate.py's old-YOLO numbers "
          "(and the skip count above against its 7/238 skipped) to see "
          "whether the smile-finetuned detector actually helped.")

    ranked = sorted(results, key=lambda r: r["dice"])
    worst = ranked[:NUM_WORST_TO_SHOW]
    best = ranked[-NUM_BEST_TO_SHOW:][::-1]

    print(f"\n--- Worst {len(worst)} ---")
    for r in worst:
        print(f"  {r['filename']}: DSC={r['dice']:.4f}  IoU={r['iou']:.4f}")

    print(f"\n--- Best {len(best)} ---")
    for r in best:
        print(f"  {r['filename']}: DSC={r['dice']:.4f}  IoU={r['iou']:.4f}")

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
