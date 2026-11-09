# Module 3c: Light HQ-SAM Zero-Shot Check

## What this module is, and isn't
This is **not** the committed architecture swap yet - it's the cheapest
possible test of whether swapping the segmenter is even worth the real
engineering cost (a new fine-tuning run, a new module). It runs Light
HQ-SAM's own pretrained weights, completely unmodified, with no fine-tuning
on our dataset at all. If a model that has never seen a single one of our
782 training images already produces cleaner masks than our fine-tuned
MedSAM checkpoint on the failure cases that survived cleanup, that's strong
evidence the failures are about boundary-quality architecture, not about
needing more of our own data. See `docs/EXPERIMENTS.md`'s module_2b section
for the 15-image comparison (with mask cleanup) that motivated this.

## Why "Light HQ-SAM" specifically (and why we're actually running `vit_b`)
It's Rank #1 in `docs/top_10_arc_segmentation_solutions.pdf`
("YOLO-Nano -> Light HQ-SAM"). HQ-SAM (`SysCV/sam-hq`) is a SAM variant
trained specifically to produce higher-quality mask *boundaries* than the
original SAM/MedSAM - exactly the failure mode still showing up in our
15-image comparison (color leakage right at the tooth-gum edge). "Light"
refers to its `vit_tiny` variant: a TinyViT/MobileSAM backbone built for
speed and edge deployment.

**In practice, `vit_tiny` isn't loadable through the pip package.** A real
run hit `KeyError: 'vit_tiny'` on `sam_model_registry` - that model type
needs separate setup code the `segment-anything-hq` pip package doesn't
ship (see `github.com/SysCV/sam-hq/issues/130`, unresolved upstream as of
this writing). Rather than vendor that code ourselves for a cheap zero-shot
sanity check, `HQSAM_MODEL_TYPE` is `vit_b` instead - same backbone size as
our fine-tuned MedSAM checkpoint, so this is still a fair "does HQ-SAM's
boundary-focused training beat MedSAM's" test. The size-for-edge-deployment
question only matters at the later distillation stage, not this check - if
HQ-SAM's approach wins here, `vit_tiny` can be revisited (with proper repo
setup) specifically for that stage, not blocking this comparison.

## Why this needs its own fresh Kaggle session
`pip install segment-anything-hq` installs under the same top-level
`segment_anything` module name as the original `pip install segment-anything`
package module_3 uses (confirmed via PyPI) - they cannot both be imported
correctly in one environment. Run this module in a separate notebook,
installing only `segment-anything-hq` (+ `ultralytics` for the YOLO box
prompt) - not `monai segment-anything`.

## Install from GitHub source, not PyPI
The PyPI package (`segment-anything-hq` v0.3) is a stale snapshot - its
`build_sam.py` still builds the plain (non-HQ) `MaskDecoder`, not
`MaskDecoderHQ`. A real run confirmed this: loading the official
`sam_hq_vit_b.pth` checkpoint into that stale package threw
`RuntimeError: Unexpected key(s) in state_dict` for every HQ-specific layer
(`hf_token`, `hf_mlp`, `compress_vit_feat`, `embedding_encoder`,
`embedding_maskfeature`) - the checkpoint expects the current architecture,
the old pip wheel doesn't have it. Install from source instead:
```
!pip uninstall -y -q segment-anything-hq segment-anything
!pip install -q "git+https://github.com/SysCV/sam-hq.git" ultralytics
```

## Reusing the trained YOLO detector, not the generic box
This experiment is deliberately scoped to change **only** the segmenter.
Localization was already validated separately in module_2b/module_3 - so
this module loads that same trained `best.pt` and skips any image where
YOLO finds nothing, rather than falling back to the old generic box, to
keep the comparison to fine-tuned-MedSAM's YOLO-prompted results as close
to apples-to-apples as possible.

## `hq_token_only`
HQ-SAM's predictor can return either its HQ-refined output alone
(`hq_token_only=True`, their recommendation for a single isolated object)
or a blend with the standard SAM output (`False`, their recommendation for
cluttered/multi-part scenes). A full dental arch is many teeth in one mask -
closer to their "multi-part" case - so this defaults to `False`.

## What happens next depending on the result
- **Light HQ-SAM's zero-shot masks are clearly cleaner** on the cases that
  beat MedSAM + cleanup (`8764.jpg`, `IMG_2385.jpg`) -> proceed to actually
  fine-tuning it on our dataset (a real module, real Kaggle training time),
  since even pretrained-only it already outperforms our fitted MedSAM.
- **Comparable or worse** -> the problem may not be MedSAM's architecture at
  all; worth revisiting boundary-aware loss (rank #6, fine-tune MedSAM
  itself with a boundary-focused loss term) before spending more effort on
  a full segmenter swap.
- **Not yet run** - this module is written but unexecuted.
