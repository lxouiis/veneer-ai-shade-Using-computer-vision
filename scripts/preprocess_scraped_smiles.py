#!/usr/bin/env python3
"""
scripts/preprocess_scraped_smiles.py
Preprocesses scraped iStock face/smile images for the veneer shade pipeline:
  1. Detects faces using YuNet (cv2.FaceDetectorYN — works on OpenCV 4.8+ and 5.x)
  2. Burns out (blacks out) eyes and upper forehead — removes gaze/distraction
  3. Crops the lower smile arc (lips, teeth, perioral skin) to 224×224
  4. Saves clean images to data/clean_smiles/ + writes dataset.csv (is_new=True)

Run after: python3 scripts/scrape_istock_smiles.py
"""

import csv
import logging
from pathlib import Path
import numpy as np
import urllib.request
import os

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("smile_preprocessor")

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "scraped_smiles"
CLEAN_DIR     = PROJECT_ROOT / "data" / "clean_smiles"
MODEL_PATH    = PROJECT_ROOT / "data" / "face_detection_yunet_2023mar.onnx"
MODEL_URL     = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# --------------------------------------------------------------------------
# TUNEABLE CONSTANTS
# --------------------------------------------------------------------------
OUTPUT_SIZE   = (224, 224)

# Fraction of image height to black out above the detected eye line
# (keeps forehead + top of eyes burned). Raise to 0.55 for tighter crops.
EYE_BURN_UPPER_FRACTION  = 0.50

# Fraction of face height below which we start keeping facial content
# (the mouth / smile zone starts here for typical portrait proportions)
SMILE_START_FRACTION     = 0.48   # from top of face bounding box
SMILE_END_FRACTION       = 1.00   # all the way to chin

# Horizontal padding to include cheek skin on each side (0.10 = 10% of width)
SMILE_HORIZ_PADDING      = 0.08


# --------------------------------------------------------------------------
# STEP 1 — DOWNLOAD FACE MODEL IF MISSING
# --------------------------------------------------------------------------
def ensure_face_model():
    if MODEL_PATH.exists():
        return
    logger.info("Downloading YuNet face detection model (~232 KB)...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(str(MODEL_URL), str(MODEL_PATH))
    logger.info(f"Model saved to {MODEL_PATH}")


# --------------------------------------------------------------------------
# STEP 2 — FACE DETECTION (OpenCV 5 FaceDetectorYN)
# --------------------------------------------------------------------------
def detect_faces(image_bgr):
    """
    Returns list of (x, y, w, h) face bounding boxes using YuNet.
    Falls back to an empty list if no face is found.
    """
    h, w = image_bgr.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH),
        "",                 # config (empty = use defaults)
        (w, h),             # input size must match the image
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5,
    )
    _, detections = detector.detect(image_bgr)
    if detections is None:
        return []
    # Each row: [x, y, w, h, landmarks×10, score]
    boxes = []
    for det in detections:
        fx, fy, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        boxes.append((fx, fy, fw, fh))
    return boxes


# --------------------------------------------------------------------------
# STEP 3 — EYE / UPPER-FACE BURNOUT
# --------------------------------------------------------------------------
def burn_out_upper_face(image_bgr, faces):
    """
    Blacks out the upper half of each detected face so only the
    lower smile/lip region is left intact.
    Falls back to blacking out top 35% of the frame if no face is found.
    """
    result = image_bgr.copy()
    ih, iw = image_bgr.shape[:2]

    if not faces:
        # No face detected — burn top 35% globally
        result[: int(ih * 0.35), :] = 0
        return result

    # Pick the largest face
    fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])

    # Black out from the top of the face down to the eye-line midpoint
    burn_y_end = fy + int(fh * EYE_BURN_UPPER_FRACTION)
    burn_y_end = min(burn_y_end, ih)
    result[:burn_y_end, :] = 0

    return result


# --------------------------------------------------------------------------
# STEP 4 — SMILE-ARC CROP
# --------------------------------------------------------------------------
def extract_smile_crop(image_bgr, faces, target_size=OUTPUT_SIZE):
    """
    Crops the lower facial zone (lips, teeth, perioral skin) from the image.
    Uses face bbox to locate the smile zone precisely.
    Falls back to bottom 55% center crop if no face is detected.
    """
    ih, iw = image_bgr.shape[:2]

    if faces:
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        y1 = fy + int(fh * SMILE_START_FRACTION)
        y2 = fy + int(fh * SMILE_END_FRACTION)
        pad_x = int(iw * SMILE_HORIZ_PADDING)
        x1 = max(0, fx - pad_x)
        x2 = min(iw, fx + fw + pad_x)
    else:
        # Fallback: bottom-centre zone
        y1 = int(ih * 0.45)
        y2 = int(ih * 0.95)
        x1 = int(iw * 0.12)
        x2 = int(iw * 0.88)

    y1, y2 = max(0, y1), min(ih, y2)
    x1, x2 = max(0, x1), min(iw, x2)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        # Guard: if crop is empty use centre half
        crop = image_bgr[ih // 4 : 3 * ih // 4, iw // 4 : 3 * iw // 4]

    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    if not CV2_OK:
        logger.error("OpenCV not installed. Run: pip3 install opencv-python-headless")
        return

    ensure_face_model()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")))
    if not images:
        logger.error(f"No images found in {RAW_DIR}. Run scrape_istock_smiles.py first.")
        return
    logger.info(f"Found {len(images)} raw images to preprocess.")

    out_records = []
    face_found_count = 0

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read {img_path.name} — skipping.")
            continue

        # Detect faces
        faces = detect_faces(img)
        if faces:
            face_found_count += 1

        # Burn out upper face / eyes
        burned = burn_out_upper_face(img, faces)

        # Extract smile crop
        smile_crop = extract_smile_crop(burned, faces)

        # Save
        clean_filename = f"clean_{img_path.name}"
        clean_path = CLEAN_DIR / clean_filename
        cv2.imwrite(str(clean_path), smile_crop)

        status = "face✓" if faces else "no-face (fallback)"
        logger.info(f"[{i}/{len(images)}] {img_path.name} → {clean_filename}  [{status}]")

        out_records.append({
            "filename":      clean_filename,
            "original_file": img_path.name,
            "is_new":        True,
            "source":        "istock_scraped",
            "face_detected": bool(faces),
        })

    # Write dataset manifest
    manifest_out = CLEAN_DIR / "dataset.csv"
    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "original_file", "is_new", "source", "face_detected"]
        )
        writer.writeheader()
        writer.writerows(out_records)

    logger.info(
        f"\n{'='*55}\n"
        f"  Processed : {len(out_records)} images\n"
        f"  Face found: {face_found_count}/{len(out_records)}\n"
        f"  Saved to  : {CLEAN_DIR}\n"
        f"  Manifest  : {manifest_out}\n"
        f"{'='*55}"
    )


if __name__ == "__main__":
    main()
