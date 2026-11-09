# Module 3: MedSAM Inference Notes

## What this module does
Module 2 produced a fine-tuned MedSAM checkpoint. This module loads that
checkpoint and uses it on **new** photos it has never seen — running the
model "forward only" (no training) to predict a tooth mask, then using that
mask to cut out a background-removed tooth crop.

## Training vs. inference
Training adjusts the model's weights to reduce error, using known
ground-truth masks. Inference is just using the already-trained model to make
a prediction on new data, with the weights left untouched
(`model.eval()` + `torch.no_grad()`). Inference is faster and doesn't need
labels — which is exactly the real-world situation (a new patient photo has
no ground-truth tooth mask).

## Why the bbox prompt is different here
During training, we derived a tight bounding box directly from the known
ground-truth mask. At inference, there is no ground-truth mask to derive a
box from — that's the whole point of running the model. The original
approach was a generous, centered default box covering most of the image
(`get_bbox_prompt`), assuming the input photo is already roughly framed
around the mouth/teeth. Reviewing real output showed this let gum/lip tissue
leak into the predicted mask — the box was too loose relative to the tight,
jittered-but-still-close boxes MedSAM actually trained on.

`get_bbox_prompt_yolo` replaces it with a real, per-image box from a small
YOLO detector (module_2b), controlled by `USE_YOLO_PROMPT`. It falls back to
the old generic box automatically — either for the whole run if the YOLO
checkpoint isn't found, or per-image if YOLO detects nothing on a particular
photo — so this is a safe toggle, not an all-or-nothing rewrite. See
`module_2b_yolo_localization/notes.md` for why YOLO was chosen over a
face-landmark approach, and `docs/DISCUSSION_next_course_of_action.md` for
the full reasoning.

## From logits to a binary mask
The mask decoder outputs raw scores ("logits"), not probabilities. We apply
`sigmoid()` to turn them into values between 0 and 1 (a probability that each
pixel is tooth), then threshold at `MASK_THRESHOLD` to get a clean binary
mask: 1 = tooth, 0 = not tooth. Raised from 0.5 to 0.9 after a real 15-image
comparison showed 0.9 gives noticeably less gum/lip contamination (more
fragmented masks, but fragmentation barely affects module 5's CIELab
averaging while gum contamination corrupts it directly) — see
`docs/DISCUSSION_segmentation_next_steps.md`.

## Cleaning up the mask before use
Reviewing real YOLO-prompted output showed two more failure shapes even with a
good box: noisy/fragmented masks (small stray blobs, tiny holes) and, on hard
images (inflamed gums, low contrast), genuine color-leakage where gum pixels
get classified as tooth. `cleanup_mask` fixes the first kind cheaply, with no
retraining: a morphological close fills small internal gaps, then connected-
component analysis drops components much smaller than the largest one found.
It deliberately does **not** keep only the single largest component - a
healthy anterior tooth row is naturally several separate components (each
tooth, with gaps between them), so that would delete real teeth, not noise.
This cannot fix the second failure shape (color-leakage) - that's a pixel
classification problem, not a shape problem, and points at the segmentation
model's boundary quality rather than the mask's geometry. See
`docs/EXPERIMENTS.md`'s module_2b section for the real 10-image comparison
that surfaced both failure shapes.

## Adaptive Interior Enamel ROI
`get_interior_enamel_mask` is a second, separate mask transform from
`cleanup_mask` above - cleanup fixes the mask's *shape* (noise blobs, small
holes), this fixes what gets *read from* the mask for color. It uses
`cv2.distanceTransform` to find, per pixel, the distance to the nearest edge
of its own blob, then keeps only pixels above `INTERIOR_ENAMEL_FRACTION` of
that *blob's own* max distance - computed per connected component, not one
global threshold, so a thin lateral incisor and a wide central incisor each
get proportional erosion instead of the widest tooth in the photo deciding
the cutoff for every other tooth. This module doesn't call it itself (its
own crop/visualization uses the full mask); module_5's `extract_cielab` uses
it specifically, since it's a color-reading concern, not a segmentation one.

## Background removal
Once we have a binary mask, "extracting the tooth crop" just means
multiplying the original image by the mask: any pixel where the mask is 0
gets zeroed out (turned black), leaving only the tooth region visible. The
mask has to be resized from MedSAM's 1024×1024 working resolution back down
to the original photo's resolution first (using nearest neighbor, for the
same reason described in module_1's notes — masks are label maps, not
photos).

## What the output is and where it goes next
Tooth crop PNGs (background removed) and comparison figures. Module 5's
integration pipeline calls the same functions used here to get the tooth
crop, then extracts its CIELab color values.
