# Module 5: Integration & CIELab Extraction Notes

## What this module does
This module doesn't introduce a new model — it **chains together** modules 3
(MedSAM tooth crop) and 4 (BiSeNet skin/lip crops) into one pipeline per
patient photo, then extracts color values from all three crops and saves
everything in an organized, per-patient folder. This is "Stage 1" of the
overall project: turning a raw photo into the 9 numbers (plus the tooth
image) that the later ViT+CLIP shade classifier will use.

## What is CIELab, and why not just use RGB?
RGB describes color the way a screen displays it (how much red/green/blue
light to emit), but it doesn't match how humans *perceive* color differences
well — equal RGB distances don't look like equally different colors to the
eye. CIELab (also written "Lab" or "L\*a\*b\*") was specifically designed to
be **perceptually uniform**: equal distances in Lab space correspond to
roughly equal perceived color differences. That property matters a lot for
matching a tooth photo to a VITA shade guide, since dental shade matching is
fundamentally a perceptual color-matching task.

CIELab has three channels:
- **L** (lightness): 0 = black, 100 = white.
- **a**: green (negative) to red (positive).
- **b**: blue (negative) to yellow (positive).

Teeth, skin, and lips each have a characteristic Lab signature, which is why
we extract all three, not just the tooth — skin and lip tone provide context
that helps the shade classifier account for lighting and individual
variation in the photo.

## Why we exclude background pixels from the average
Every crop (tooth, skin, lip) is produced by zeroing out everything outside
the segmented region, so most crops are mostly black background with only a
small region of actual tooth/skin/lip pixels. If we naively averaged the
*entire* crop, the huge black background would drag the average lightness
(L) way down toward 0, producing a meaningless "color." Instead,
`extract_cielab` only averages over the pixels the mask actually marked as
foreground, so the numbers reflect the real tooth/skin/lip color.

## Reading tooth color from the interior, not the whole mask
Real inference output showed that even a decent tooth mask still leaks some
gum color right at its own boundary - averaging color over the *entire* mask
lets those edge pixels pull the reading toward pink/red, even when the mask
"looks fine" at a glance. `tooth_color_mask` (from module_3's
`get_interior_enamel_mask`) erodes each tooth blob inward before this
pipeline reads its color, keeping only the pixels safely away from the
boundary. This is a downstream color-extraction fix, not a segmentation fix
- it doesn't make the mask itself any more accurate, it just stops reading
color from the part of the mask most likely to be wrong. See
`module_3_medsam_inference/notes.md` and `docs/EXPERIMENTS.md`'s module_2b
section for the real comparison that motivated this.

## OpenCV's LAB encoding vs. true CIELab
OpenCV stores 8-bit LAB images with all three channels squeezed into the
0-255 range for storage reasons: L is scaled to 0-255 (not 0-100), and a/b
are shifted up by 128 so they fit in an unsigned byte (instead of their true
roughly -128 to 127 range). Before using the values, we convert back to true
CIELab units: `L = L_cv × 100/255`, `a = a_cv − 128`, `b = b_cv − 128`.
Skipping this step would silently produce numbers on the wrong scale.

## Graceful degradation
Not every photo will contain a clean skin or lip region (for example, a
close-cropped tooth photo). Rather than crash, this pipeline records `NaN`
for that region's Lab values and adds a flag like `"skin_region_not_found"`
or `"lip_region_not_found"` to `meta.json`, so downstream code (or a human)
can see exactly what happened for that patient.

## What the output is and where it goes next
For each patient: `tooth.png`, `skin.png`, `lip.png`, and `meta.json`
(9-number color vector, bbox, mask pixel areas, flags). The 9-number color
vector plus the tooth crop image are exactly the inputs the later ViT+CLIP
shade classification model (not part of modules 0-5) uses to predict a VITA
shade label like `"A2"`.

## Preferring Doc Sans's reviewed masks over a fresh prediction
`APPROVED_MASKS_DIR` points at `doctor_review_app`'s human-verified tooth
masks (synced locally via its `download_approved_masks.py`, then published
as a Kaggle Dataset the same way the checkpoints are). When a reviewed mask
exists for an image, `run_stage1_pipeline` uses it directly instead of
calling MedSAM - a doctor-checked mask is a strictly more trustworthy color
source than raw model output, which matters most if this output ends up as
training data for `module_6_vit_clip`'s pseudo-labeling, not just a one-off
reading. Falls back to a fresh YOLO+MedSAM prediction automatically for any
image not yet reviewed (or if `APPROVED_MASKS_DIR` doesn't exist at all) -
`meta.json`'s flags record which source was actually used
(`tooth_mask_doctor_approved` vs `tooth_mask_model_predicted`) per patient,
so this is traceable later, not silently mixed together.

## Why SAMPLE_IMAGES_DIR changed from CelebAMask-HQ to NCT
The original `SAMPLE_IMAGES_DIR` pointed at CelebAMask-HQ as a cheap smoke
test, explicitly because BiSeNet needs a face in frame and CelebA guarantees
one - it was never meant to be the real input. The real input is NCT itself:
the same photo gets read by MedSAM (tooth) and BiSeNet (skin/lip) in one
pass. Many NCT photos are tight intraoral close-ups with no visible face -
those still run without crashing, they just pick up
`skin_region_not_found`/`lip_region_not_found` flags and `NaN` in that part
of the color vector, same graceful degradation as above.
