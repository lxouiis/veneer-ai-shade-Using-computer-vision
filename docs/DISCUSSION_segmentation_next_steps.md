# Discussion: MedSAM segmentation quality & next course of action

Date: 2026-09-05
Context: reviewing real Kaggle inference output (`infer.py`, 15 val samples) after
the 0.8838-DSC checkpoint, deciding what to do about visibly imperfect masks.

## 1. What we actually found in the notebooks

Three notebooks were pasted into `docs/`: `notebook-c (0.5).ipynb`,
`notebook-c.(0.9).ipynb`, and `notebook-c.ipynb`. First pass turned up a mixup:
the file named `(0.9)` still had `MASK_THRESHOLD = 0.5` in its source cell — it
was never actually run at 0.9, so its output was byte-identical to the 0.5 run.
`notebook-c.ipynb` (no suffix) is the real 0.9 run — confirmed
`MASK_THRESHOLD = 0.9`, same 15 images, same random seed, genuinely comparable.

## 2. Real 0.5 vs 0.9 comparison

Same 15 val images, same seed, threshold as the only variable.

**At 0.5**: masks are solid blobs that reliably cover the tooth region but
bleed into gum and lip tissue at the top/bottom edges — often a visible pink
strip along the gumline gets included as "tooth."

**At 0.9**: masks break into separated per-tooth vertical strips that hug the
actual tooth boundaries much more tightly, with far less gum/lip pink pulled
in. The tradeoff is more internal fragmentation — small holes and gaps inside
each tooth, some tiny disconnected blobs.

**Read for our use case**: fragmentation barely matters for module 5's job
(averaging LAB over whatever pixels are in the mask — a hole just means a few
fewer tooth pixels averaged, still valid). Gum contamination matters a lot —
pink gum pixels pull the averaged tooth `a` (green-red) and `L` values away
from true enamel color, directly corrupting the one thing module 5 exists to
measure. On that basis, **0.9 looks like the better operating point for us**,
not 0.5. This is a real, cheap, no-retrain win — just change the constant.

This doesn't fix the underlying cause, though: the model is genuinely unsure
about tooth/gum boundary pixels (that's why raising the threshold changes the
mask meaningfully). Threshold tuning trims the fuzz; it doesn't teach the
model where the boundary actually is.

## 3. Root cause: the box prompt, not the model

Across all 15 samples, `get_bbox_prompt()` is a generic box shaved 10% off
each edge of the whole image — it has no idea where the mouth actually is in
a given photo. Any tooth-adjacent tissue inside that generic box is fair game
for the model to include. Raising the DSC ceiling on this checkpoint, or
swapping architectures, does not fix a bad prompt — a sharper model given a
loose box will still be tempted to over-include nearby tissue.

## 4. Is MedSAM the right model at all?

MedSAM/SAM is a *promptable* segmenter — built for "point/box in, mask out,"
designed for interactive multi-object use. We only ever want one class
(tooth), always the same class, no interactivity. That mismatch is structural,
not a tuning problem:
- We pay for box-prompt machinery we don't need and that actively hurts us
  when the box is generic.
- A plain fully-supervised segmenter (U-Net, DeepLabV3+) trained directly on
  our 782 mask pairs has no prompt step at all — it looks at the photo and
  predicts a mask, full stop. Removes this failure mode by construction.
- Cost: a new architecture, new training code, another Kaggle training run —
  not a drop-in change like the threshold or a better prompt.

**Recommendation**: don't jump to a new architecture yet. Try the two cheap
fixes below first (better prompt, threshold), because the failure mode we're
actually seeing (gum leakage) is explained by the prompt, not by MedSAM's
ceiling. If a properly-prompted MedSAM still leaks gum tissue after that,
*that's* the signal to invest in a from-scratch segmenter instead.

## 5. Smile line / smile arc — two concrete uses

Raised as a new idea this session — maps to a real clinical concept (the
curve traced by the incisal edges of the upper anterior teeth against the
lower lip). Two distinct, useful applications, not one:

**(a) As a better prompt (cheap, no retrain).** Run a face-landmark model
(e.g. MediaPipe FaceMesh, has mouth/lip landmark indices) once per photo to
locate the actual mouth opening, and derive the SAM box from those landmarks
instead of a fixed 10%-margin rectangle. This directly targets the root cause
in §3 — the box would finally reflect where the mouth is in *this* photo, not
a fixed sliver of image geometry. Also naturally handles zoomed-out full-face
photos vs close-up intraoral shots differently, which the current fixed-margin
box can't.

**(b) As a QA/evaluation metric (novel, addresses a real gap).** We have no
ground truth mask at deployment time — DSC is uncomputable once a real patient
photo comes in, only Kaggle val images have it. Fitting a smooth curve to the
top edge of a predicted mask and scoring deviation from an expected arc shape
gives an automatic, reference-free plausibility check: a mask that ignores the
arc (e.g. one blown out toward gum tissue, or missing teeth at the corners)
scores badly on arc-fit even with no ground truth to compare against. This
would let module 3 flag "this segmentation looks wrong" per-image at
inference time, which we currently have no way to do.

## 6. Proposed order of operations

1. **Set `MASK_THRESHOLD = 0.9`** in `infer.py` — one-line change, supported
   directly by the comparison in §2.
2. **Landmark-derived box prompt** — replace `get_bbox_prompt()`'s fixed
   10%-margin box with one derived from MediaPipe FaceMesh mouth landmarks.
   No retraining; changes only module 3.
3. **Re-run `infer.py`** on the same 15-image sample with both changes and
   compare against this baseline — check whether gum leakage is actually
   gone, not just thinner.
4. **Smile-arc QA metric** — implement as a scoring function in module 3
   alongside the mask output; log the score into `meta.json` in module 5
   rather than gating anything yet.
5. **Only if step 3 still shows leakage**: revisit whether MedSAM is the
   right architecture at all, and consider a from-scratch U-Net/DeepLabV3+
   trained on our 782 pairs as a real alternative (§4) — bigger lift, new
   training run, but removes the prompt-dependency failure mode entirely.

No code has been changed yet — this file is the discussion record. Next step
is confirming whether to start on item 1/2 above.
