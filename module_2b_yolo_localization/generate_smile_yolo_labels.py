import os
import numpy as np
import cv2

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# SmileImages has no YOLO .txt labels (only preprocessed image/mask .npy
# pairs from the MedSAM side of the project) - the NCT dataset's own labels
# were hand-annotated. Rather than hand-annotate SmileImages too (no time),
# this derives boxes directly from its existing GT masks: each connected
# component of the mask becomes one YOLO box, matching the NCT label
# convention confirmed in module_0's exploration (one box per tooth, not one
# box for the whole arch) - training on a different box granularity than the
# existing data would teach the detector two inconsistent conventions.
SMILE_DATA_DIR = "/kaggle/input/datasets/hemantdarur/smileimages/processed"
OUTPUT_DIR = "/kaggle/working/yolo_smile_data"

# Same noise-filtering convention as cleanup_mask elsewhere in this project -
# drops components far smaller than the largest one found per image (mask
# speckle, not real teeth) rather than keeping only the single largest
# component (a tooth row is legitimately several separate components).
MIN_COMPONENT_FRACTION = 0.01


# --------------------------------------------------------------------------
# MASK -> YOLO BOXES
# --------------------------------------------------------------------------
def mask_to_yolo_lines(mask):
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return []

    max_area = stats[1:, cv2.CC_STAT_AREA].max()
    min_area = max_area * MIN_COMPONENT_FRACTION

    lines = []
    for label_id in range(1, num_labels):
        x, y, box_w, box_h, area = stats[label_id]
        if area < min_area:
            continue
        center_x = (x + box_w / 2) / w
        center_y = (y + box_h / 2) / h
        norm_w = box_w / w
        norm_h = box_h / h
        lines.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
    return lines


# --------------------------------------------------------------------------
# PER-SPLIT CONVERSION
# --------------------------------------------------------------------------
def process_split(split):
    images_dir = os.path.join(SMILE_DATA_DIR, split, "images")
    masks_dir = os.path.join(SMILE_DATA_DIR, split, "masks")
    out_images_dir = os.path.join(OUTPUT_DIR, split, "images")
    out_labels_dir = os.path.join(OUTPUT_DIR, split, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    filenames = sorted(f for f in os.listdir(images_dir) if f.endswith(".npy"))
    written, skipped = 0, 0

    for filename in filenames:
        image = np.load(os.path.join(images_dir, filename))  # uint8 HxWx3 RGB
        mask = np.load(os.path.join(masks_dir, filename))

        lines = mask_to_yolo_lines(mask)
        if not lines:
            skipped += 1  # empty mask after cleanup - no real box to teach YOLO from
            continue

        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(
            os.path.join(out_images_dir, base_name + ".jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        with open(os.path.join(out_labels_dir, base_name + ".txt"), "w") as f:
            f.write("\n".join(lines))
        written += 1

    print(f"[{split}] wrote {written} labeled images, skipped {skipped} (no usable mask)")


if __name__ == "__main__":
    process_split("train")
    process_split("val")
