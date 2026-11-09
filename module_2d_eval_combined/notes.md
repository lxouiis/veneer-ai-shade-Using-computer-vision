# Module 2d: Evaluate Finetune+YOLO on the Combined Dataset

## What this fixes
`docs/finetune.ipynb`'s own `validate()` scores the new checkpoint (resumed
on NCT + SmileImages, image encoder unfrozen) against a box derived from the
**ground-truth mask** (jittered by `BBOX_SHIFT`) - same as module_2's
original training script. That's the right box to validate *training
progress* against (apples-to-apples with earlier DSC numbers), but it is not
what happens at real inference: a new patient photo has no GT mask to derive
a box from (see `module_3_medsam_inference/notes.md`). This script re-scores
the same checkpoint using an actual YOLO-predicted box instead, so the
number you get is "how well does this checkpoint do in the real pipeline,"
not "how well does it do when handed a box computed from the answer."

## Why no new "combining" step is needed
`docs/finetune.ipynb`'s cell 3 already merged SmileImages' `.npy` files into
`/kaggle/working/processed/{train,val}/{images,masks}` next to the NCT data.
That merged folder *is* the combined dataset - this script just points at it
(`find_data_dir`) rather than redoing the merge. If you're evaluating in a
fresh Kaggle session instead of right after running finetune.ipynb, that
folder won't exist - either keep this in the same session, or publish
`/kaggle/working/processed` and `/kaggle/working/checkpoints` as a Kaggle
Dataset first and attach it (see `find_checkpoint`/`find_data_dir`'s
fallback candidates).

## The `mask_decoder` sanity check
`notebooka818589fdd.ipynb`'s install cell pulls `sam-hq`
(`git+https://github.com/SysCV/sam-hq.git`), not plain `segment-anything` -
if `docs/finetune.ipynb`'s training cell ran in that same environment,
`sam_model_registry["vit_b"]` would have built an HQ-architecture model
(`MaskDecoderHQ`, extra `hf_token`/`hf_mlp`/etc. layers) under the hood, and
`model.mask_decoder(...)`'s plain-SAM call signature (no `hq_token_only`,
no `interm_embeddings`) may not even match what that class expects. This
script prints `type(model.mask_decoder).__name__` right after loading -
it should say `MaskDecoder`. If it says `MaskDecoderHQ`, the checkpoint was
very likely fine-tuned (or is being evaluated) against the wrong decoder
architecture, and the DSC number isn't trustworthy - the fix is re-running
the fine-tune in a session with plain `pip install segment-anything` only,
the same isolation rule `module_3c_hqsam_zeroshot/notes.md` already
documents for the reverse case.

## Why DSC and IoU, not DSC alone
`docs/tooth_shade_pipeline_raw_findings.md` already flags this: "Do not use
only DSC." Both come for free from the same prediction (no extra forward
pass), so there's no real cost to reporting both, plus the median DSC as a
cheap check for whether the mean is being dragged by a handful of bad
outliers.

## Why skip rather than fall back when YOLO finds nothing
Matches `module_3c_hqsam_zeroshot`'s convention: a generic fallback box
would mix "YOLO failed" cases in with "the segmenter did badly" cases under
one averaged score, making the number harder to interpret. The skip count is
printed so you can see how often it happens - if it's a large fraction of
the SmileImages subset specifically, that's itself a signal the detector may
need retraining on smile-framing photos, not just the segmenter.

## What to do with the result
- **New DSC/IoU clearly higher than the 0.8838 baseline** (`docs/
  DISCUSSION_next_course_of_action.md`) under the *same* YOLO-box setup ->
  the encoder-unfreeze + SmileImages data genuinely helped; worth publishing
  this checkpoint as the new default and re-running module_3's own 15-image
  visual comparison to confirm gum-leakage is actually reduced, not just the
  pixel-overlap number.
- **Flat or worse** -> the earlier run 2 in `docs/EXPERIMENTS.md` (encoder
  unfrozen, no SmileImages) already saw DSC *drop* slightly (0.8838 ->
  high-0.86s) with more training; this may be the same ceiling, not a
  SmileImages-specific problem. Worth checking the `mask_decoder` sanity
  print above before concluding anything either way.
