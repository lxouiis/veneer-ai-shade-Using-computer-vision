# Module 1b: Gum/Lip Suppression Pre-Filter

## What this module is, and isn't
This is **not** another mask-cleanup step layered on at the end (that's what
`cleanup_mask` in module_2b/3/3c and the interior-enamel erosion in
module_3/5 already do). This runs first, on the raw photo, before it ever
reaches YOLO or the segmenter - it detects gum and lip tissue by colour
alone and burns those pixels to black, so the box prompt and the
segmentation model never see them as candidate "tooth" pixels in the first
place. Like module_3c, it's written as a standalone, cheap check first - run
it and look at the visualizations before wiring it into the real pipeline.

## Why this needs to happen *before* YOLO/SAM, not just after
Every fix so far (the 0.9 mask threshold, `cleanup_mask`, the interior
enamel ROI in module_3/5) works on the segmenter's *output* - they can only
patch a mask that already includes some gum. The real screenshots that
motivated this (Light HQ-SAM zero-shot, `docs/Screenshot 2026-09-12 at
11.33.42/34.00 AM.png`) show the failure isn't a few stray boundary
pixels - it's a large, deliberate scallop of pink gum tissue included as
part of the predicted mask, because YOLO's box and the segmenter's own
training both treat "the region between the box edges" as fair game. No
amount of post-hoc erosion removes that without also eating into real
tooth pixels. Removing the gum/lip signal from the *input* instead means:
- YOLO's box tightens around what's still visible (mostly teeth), instead
  of a box that includes gum because gum was never excluded from its
  training assumption of "the arch region".
- The segmenter has no gum-coloured pixels left to (mis)classify as tooth -
  there's nothing pink left in the crop for it to include.

## Why colour, not a model
There's no "gum" class in BiSeNet's 19-label face-parsing output (module_4)
- CelebAMask-HQ's labels distinguish skin/lips but not gum vs. tooth inside
an open mouth, and BiSeNet expects a full aligned face, which several of our
real images aren't (compare `patient9-2.jpg`, a tight intraoral close-up
with no face at all, to `IMG_2423.jpg`, a fuller smile-with-chin shot -
both need this, in one framing-agnostic method). A classical HSV/Lab
threshold needs no training data, runs on any framing, and gum/lip tissue's
colour signature (pink-red, saturated) versus enamel's (near-neutral,
desaturated, sometimes yellow-tinted) is one of the more reliable colour
separations in dental photography - this is the same physical property
dentists use to visually distinguish the two.

## Why Lab's `a` channel is the primary signal, not HSV hue alone
Hue is noisy exactly where it matters most: near-grey/near-white pixels
(like enamel) have unstable, meaningless hue because saturation is near
zero there - small sensor noise can swing hue anywhere. Lab's `a` channel
(green-red axis) stays meaningful at low saturation and separates
"reddish" from "neutral" directly. `GUM_LIP_A_THRESHOLD = 12` was chosen so
enamel (`a` close to 0, sometimes slightly negative under blue-ish lighting)
sits safely below it while gum/lip tissue (`a` well into positive territory)
sits above. Saturation and a value floor are still ANDed in, to avoid
flagging near-black shadow gaps between teeth (where `a` can be unreliable
too) as gum.

## The dilation step
`GUM_LIP_DILATE_PX` grows the detected gum/lip region outward by a few
pixels before burning it out, specifically to also catch the blended
transition band right at the tooth-gum edge - real photos don't have a
hard pixel boundary between the two, there's a soft gradient a few pixels
wide, and that gradient is exactly the "boundary colour-leakage" failure
mode called out in module_3's notes and `docs/EXPERIMENTS.md`. Eating a
little into that band costs a few genuinely-tooth pixels at the very edge;
leaving it in costs a strip of near-gum pixels bleeding into the reading.
Given module_5 already reads colour from an eroded *interior* enamel ROI,
not the mask edge, losing a few pixels right at the boundary is the safer
trade.

## Why burn to black, not skip the pixel some other way
Black matches the "removed" convention already used everywhere else in this
project (module_3/5's background-removed crops) - it's the same operation,
just applied earlier and to gum/lip color instead of "outside the mask".
Feeding black pixels into YOLO/SAM also gives the strongest possible signal
that there's nothing there to detect or segment, which is exactly the
intended effect.

## How to test this in isolation (do this before wiring it in)
Run `suppress_gum_lip.py` on its own, on the same 15-image sample module_3c
uses (`RANDOM_SEED = 42`), and look at the three-panel visualizations:
- Does the detected gum/lip mask (middle panel) actually track the pink
  tissue in `patient9-2.jpg`/`IMG_2423.jpg`-style photos, without eating
  large chunks of enamel (which would show up as bright white regions
  disappearing in the burned panel)?
- Check the printed `burned X% of pixels` line per image - a healthy tight
  intraoral crop should burn maybe 10-30%; something burning >60% likely
  means the thresholds are too loose for that lighting/skin tone and need
  tuning before trusting this on the real pipeline.

## How to wire this into the real pipeline once validated
Not done yet in this pass - deliberately, to keep this a cheap, reversible
check first (same reasoning as module_3c). If the visual check looks good:
1. In module_2b's `train_yolo.py` inference use and module_3/3c's
   `preprocess_image` callers: right after computing `image_resized_uint8`,
   call `get_gum_lip_mask` then `burn_out_gum_lip` on it, and pass the
   *burned* image into `get_bbox_prompt_yolo` and `predictor.set_image` -
   the untouched `original_image` stays untouched for the final crop.
2. In module_3/5's `extract_tooth_crop`/`extract_cielab`: recompute
   `get_gum_lip_mask` on `original_image` (full resolution) and AND its
   inverse into the final crop mask, as one more safety net in case the
   segmenter still included a gum pixel the burn-out missed at its own
   dilated edge.
3. Re-run the same 15-image comparison used throughout module_3c's
   experiments, before/after, and check `docs/EXPERIMENTS.md`-style whether
   gum contamination in the final tooth crop actually drops, and that
   real tooth structure (incisal edges, tips of teeth near the gumline)
   isn't being eaten along with it.

## What happens next depending on the result
- **Gum/lip cleanly removed, teeth intact** -> wire in per the steps above,
  update module_2b/3/3c/5's own notes.md to mention the new upstream step.
- **Eating into real tooth pixels (especially near the gumline, where
  enamel is naturally a bit more translucent/yellow and closer to gum
  colour)** -> tighten `GUM_LIP_A_THRESHOLD` upward and/or shrink
  `GUM_LIP_DILATE_PX` before re-testing, rather than abandoning the
  approach - the failure mode in the screenshots is large, obvious gum
  regions, well clear of that edge case.
- **Not yet run** - this module is written but unexecuted, same status as
  module_3c before its own real Kaggle run.
