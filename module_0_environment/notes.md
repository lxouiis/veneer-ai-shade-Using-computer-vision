# Module 0: Environment & Dataset Exploration Notes

## What this module does
Before writing any training code, we look at the raw dataset with our own eyes
(well, matplotlib's) and confirm three things: our libraries are installed and
working, the folder structure matches what we expect, and the images/masks/
labels actually line up correctly. Skipping this step is how you end up
debugging a training crash three modules later that was really just a wrong
file path.

## Why we check library versions
MedSAM, OpenCV, and scikit-image all behave slightly differently across
versions (e.g. resize defaults changing). Printing versions at the start means
if something breaks later, we can immediately rule in/out "it's a version
mismatch" as the cause.

## Why we walk the whole folder tree
`os.walk` prints every subfolder and a few sample filenames from each. This
catches surprises early — e.g. an extra nested folder (like `Tooth dataset
v4/`), a differently-named split, or a folder that's empty when we expected
files in it.

## Image vs. mask vs. label counts
- **Images**: the raw `.jpg` photos.
- **Masks**: `.png` segmentation maps (`imagename_mask.png`) marking which
  pixels are tooth vs. background — used to train MedSAM.
- **Labels**: `.txt` files in YOLO format giving bounding boxes — not shade
  labels, and not used for the segmentation model, but worth knowing about
  since they exist in this dataset (could be useful later for a
  detection-based crop step).

Confirmed on the actual mounted dataset: it's the reverse of what you might
guess - masks slightly *outnumber* images (804 masks vs 782 images in train;
213 vs 209 in val), because a handful of masks have no corresponding image
(orphan annotations), not the other way around. Every image does have a
matching mask. This is exactly why we always match by filename rather than
assuming any particular file-count relationship between images and masks.

## What YOLO format means
Each line in a YOLO `.txt` label file is one bounding box:
```
class_id center_x center_y width height
```
All four numeric values are normalized to the 0-1 range (a fraction of the
image's width or height), not raw pixels. This makes the label file resolution
-independent — the same `.txt` file works whether the image is 512x512 or
2048x2048, since you just multiply by the actual image dimensions to recover
pixel coordinates.

## Why we display images, masks, and overlays
- Sample images confirm the photos loaded and decoded correctly (right colors,
  not corrupted).
- Sample masks (grayscale) confirm the segmentation labels look like sane
  tooth-shaped regions, not noise.
- The overlay (mask drawn in red on top of the image) is the most important
  check: it visually confirms the mask is actually aligned with the tooth in
  the photo, not just present as a file.

## Why we only sample a subset for average dimensions
Reading every single image file just to check its size is unnecessary work.
Sampling a reasonable number (50) gives a good estimate of typical image
dimensions across the dataset without decoding all ~1000 images.

## Where outputs go
All generated figures (`sample_images.png`, `sample_masks.png`,
`overlay_check.png`) are saved to `/kaggle/working/module_0_environment/`,
since `/kaggle/input/` is read-only on Kaggle and `/kaggle/working/` is where
notebook outputs/artifacts persist.
