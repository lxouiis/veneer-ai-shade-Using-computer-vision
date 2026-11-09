import os
import glob
from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# Resumes the existing trained detector (module_2b's tooth_yolo/weights/
# best.pt) and continues training with SmileImages added, rather than
# training from yolov8n.pt again - cheaper and lower-risk with little time
# left, and keeps whatever the original 50-epoch NCT run already learned.
NCT_DATASET_ROOT = "/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4"
# From generate_smile_yolo_labels.py - must run in the same session, or
# published as a Dataset and attached first.
SMILE_YOLO_DATA_DIR = "/kaggle/working/yolo_smile_data"

OUTPUT_DIR = "/kaggle/working/yolo_checkpoints_smile"
DATA_YAML_PATH = os.path.join(OUTPUT_DIR, "tooth_data_combined.yaml")


def find_existing_checkpoint():
    candidates = [
        "/kaggle/input/datasets/hemantdarur/yolomodel-inference/yolo_checkpoints/tooth_yolo/weights/best.pt",
    ]
    candidates.extend(glob.glob("/kaggle/input/**/tooth_yolo/weights/best.pt", recursive=True))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


BASE_CHECKPOINT = find_existing_checkpoint()

# Ultralytics accepts a list of paths per split, so NCT and SmileImages can
# train together without physically merging folders (lower risk of
# accidentally duplicating/overwriting files across two separate datasets).
DATA_YAML_CONTENT = f"""\
train:
  - {NCT_DATASET_ROOT}/train/images
  - {SMILE_YOLO_DATA_DIR}/train/images
val:
  - {NCT_DATASET_ROOT}/val/images
  - {SMILE_YOLO_DATA_DIR}/val/images
names:
  0: tooth
"""

RUN_NAME = "tooth_yolo_smile"
IMG_SIZE = 1024   # match module_1/module_2's working resolution
EPOCHS = 30       # fewer than the original 50 - this resumes a trained
                  # detector rather than starting from yolov8n.pt
BATCH_SIZE = 16
RANDOM_SEED = 42

DEVICE = 0  # ultralytics: 0 = first CUDA GPU; falls back to "cpu" automatically if none available


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DATA_YAML_PATH, "w") as f:
        f.write(DATA_YAML_CONTENT)

    weights_to_load = BASE_CHECKPOINT if BASE_CHECKPOINT else "yolov8n.pt"
    print(f"Resuming YOLO fine-tune from: {weights_to_load}")
    if BASE_CHECKPOINT is None:
        print("Warning: no existing tooth_yolo checkpoint found - training "
              "from the base yolov8n.pt instead, same as the original run.")

    model = YOLO(weights_to_load)
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=OUTPUT_DIR,
        name=RUN_NAME,
        seed=RANDOM_SEED,
        device=DEVICE,
    )

    best_path = os.path.join(OUTPUT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"Best checkpoint saved to: {best_path}")
    print("Publish this as a Kaggle Dataset, then point module_3/3c/5's "
          "YOLO_CHECKPOINT_PATH at it to use the smile-aware detector.")
