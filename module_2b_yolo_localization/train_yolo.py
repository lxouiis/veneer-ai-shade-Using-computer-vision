import os
from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
DATASET_ROOT = "/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4"
OUTPUT_DIR = "/kaggle/working/yolo_checkpoints"
DATA_YAML_PATH = os.path.join(OUTPUT_DIR, "tooth_data.yaml")

# Ultralytics finds each image's labels by swapping "images" for "labels" in
# its path - exactly our dataset's existing train/images + train/labels (and
# val/images + val/labels) layout, confirmed in module_0's exploration. No
# reformatting needed, only this YAML pointing at the dataset root.
DATA_YAML_CONTENT = f"""\
path: {DATASET_ROOT}
train: train/images
val: val/images
names:
  0: tooth
"""

BASE_MODEL = "yolov8n.pt"  # "n" = nano: smallest/fastest YOLOv8 size, matching
                            # the "YOLO-Nano" idea from docs/top_10_arc_segmentation_solutions.pdf
RUN_NAME = "tooth_yolo"
IMG_SIZE = 1024   # match module_1/module_2's working resolution
EPOCHS = 50
BATCH_SIZE = 16   # YOLO has no SAM-style batch=1 decoder constraint - a normal detector batch
RANDOM_SEED = 42

DEVICE = 0  # ultralytics: 0 = first CUDA GPU; falls back to "cpu" automatically if none available


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DATA_YAML_PATH, "w") as f:
        f.write(DATA_YAML_CONTENT)

    model = YOLO(BASE_MODEL)
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
    print("Publish this as a Kaggle Dataset (same Notebook A -> Dataset -> "
          "Notebook B handoff pattern used for medsam_tooth_best.pth) so "
          "module_3_medsam_inference/infer.py can load it.")
