# Experiment & Decision Log

Chronological record of what was tried, what happened, and why — not conceptual
explanations (those live in each module's `notes.md`), just the actual history.
Kaggle sessions/containers reset, so this is the durable record.

## Environment / platform issues

- **P100 accelerator failed**: `CUDA error: no kernel image is available`. Root
  cause: PyTorch dropped wheel support for `sm_60` (Pascal/P100) — only
  `sm_70+` (T4 and newer) is supported. Not fixable in code. Switched to T4.
- **`ModuleNotFoundError: No module named 'monai'`** recurred multiple times.
  Cause: any Kaggle accelerator change or new session resets the container,
  wiping installed packages. Fix each time: re-run
  `!pip install -q monai segment-anything` before `train.py`.
- **"Save Version" failing on Kaggle**: caused by a GPU conflict (interactive
  session still running) when trying "Save & Run All (Commit)" after
  developing interactively. Resolved by fully committing to the Save & Run
  All workflow and adding `MAX_TRAINING_SECONDS` as a safety net so an
  unattended commit run always finishes cleanly instead of getting killed
  mid-epoch.
- **Declined**: using a second Kaggle account to get a second 30hr/week GPU
  quota. This is multi-accounting to bypass usage limits, against Kaggle's
  ToS, with ban risk to the linked/main account. Not pursued.

## module_1 preprocessing

- `skimage.transform.resize` (`anti_aliasing=True`) took ~23 minutes for 782
  images — its Gaussian anti-aliasing pass is pure Python/SciPy and very slow.
  Switched to `cv2.resize` (`INTER_CUBIC` for images, `INTER_NEAREST` for
  masks): same 782 images in ~22 seconds.
- Pair-count mismatch (782 matched pairs vs. 804/213 mask files) investigated
  with a throwaway `diagnose_pairs.py` script. Confirmed **not a bug** — 0
  unmatched images, just orphan mask annotations with no matching image (22 in
  train, 4 in val). Real counts: train 782 images / 804 masks, val 209 images
  / 213 masks.
- Switched saved arrays from float32 to uint8 (normalization moved into the
  training `Dataset.__getitem__` instead) — cut output from ~12.8GB to
  ~3.2GB, comfortably under Kaggle's `/kaggle/working` budget.

## module_2 MedSAM fine-tuning — training runs

| # | Config | Result | Notes |
|---|--------|--------|-------|
| 1 | Frozen image+prompt encoder, no augmentation, 100 epochs, `BATCH_SIZE=1`, AMP on | **Best val DSC 0.8838** | First full run. Plateaued in the high-0.80s, well below the base paper's reported 0.97. This is the current best checkpoint (`medsam_tooth_best.pth`, published as Kaggle Dataset `notebook-a-meowmeow`). |
| 2 | Resumed from run 1, encoder unfrozen (`FREEZE_IMAGE_ENCODER=False`, `ENCODER_LEARNING_RATE=1e-5`, 10x smaller than decoder LR), augmentation on (flip + brightness/contrast), 20 epochs | Val DSC at epochs 5/10/15/20: 0.8727 / 0.8782 / 0.8701 / 0.8763 — **never beat the resumed baseline (~0.8838)** | ~453s/epoch vs. ~188s/epoch frozen (2.4x slower). Checkpoint file was never overwritten since no epoch beat the starting `best_dsc` — the saved `.pth` is still run 1's weights, nothing was lost. Trajectory was still noisy/non-monotonic when the run ended, so inconclusive whether more epochs would eventually help or whether this is a real ceiling for this setup. |

**Architectural constraint discovered**: `BATCH_SIZE` must be exactly 1. SAM's
`mask_decoder.predict_masks()` does
`torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)`, built for
"1 image, N prompts" — not "N images, 1 prompt each" (our training shape).
`BATCH_SIZE=2` produced a `RuntimeError: size 4 vs 2` deep inside the decoder.
Not fixable by raising batch size; gradient accumulation is the correct
workaround (proposed, not yet implemented).

**Current decision**: not chasing DSC further for now. Run 1's checkpoint
(0.8838, frozen encoder) is the working baseline. Next step is validating it
end-to-end through module 3 → 5 to see whether segmentation quality is
actually the pipeline's bottleneck before investing more GPU time here. See
"Open questions / next steps" below for the prioritized plan.

## module_4 BiSeNet face parsing

- Used out-of-the-box (CelebAMask-HQ pretrained weights, `79999_iter.pth`),
  not fine-tuned — the reasoning: it already segments skin/lips reliably on
  the general face domain we need (skin, upper lip, lower lip), and we have
  no labeled dataset of our own faces to fine-tune it on.
- `SAMPLE_IMAGES_DIR` corrected twice: placeholder guess
  `/kaggle/input/face-photos` → unverified guess
  `/kaggle/input/celebamaskhq/CelebA-HQ-img` → confirmed real path
  `/kaggle/input/datasets/ipythonx/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img`.
- **Verified working**: 3 test images (28855.jpg, 13280.jpg, +1 more) all
  showed `skin_found=True, lip_found=True` with clean parsing maps and crops
  (confirmed via user screenshot).
- `test_robustness.py` written to spot-check harder cases (Blurry,
  Eyeglasses, Wearing_Hat attributes from `CelebAMask-HQ-attribute-anno.txt`)
  instead of random clean samples. **Not yet run** — the attribute file
  format is assumed (standard CelebA format: line 0 = count, line 1 =
  attribute names, rest = `filename val1 val2 ...`) but unverified against
  the real file.

## module_3 / module_5 — inference & integration

- `CHECKPOINT_PATH` in `infer.py` finalized to the real handoff path once the
  Kaggle Dataset was published from Notebook A's output:
  `/kaggle/input/datasets/hemantdarur/notebook-a-meowmeow/checkpoints/medsam_tooth_best.pth`.
- `pipeline.py`'s CIELab extraction masks out background pixels before
  averaging (never averages the whole crop, which would drag L toward 0 from
  the zeroed-out background) and rescales OpenCV's 8-bit LAB encoding to true
  CIELab ranges.
- **Not yet run end-to-end** — this is the current next step.

## Base paper comparison (2026-09-04)

Paper identified and saved: `docs/Full Length Paper.pdf` — Efitli, Karcioglu, Ozdogan,
Karatas, Senocak, *"Tooth color prediction in intraoral images under different
clinical lights using ML algorithms and CLAHE technique: an In-Vivo study,"*
Lasers in Medical Science (2025) 40:411.

**Correction to earlier framing**: the paper's 97% is classification accuracy
(RF/XGBoost, 16-class Vita-style shade code) on already-isolated tooth ROIs —
not a segmentation DSC. It is not a fair comparison to module 2's 0.8838
segmentation Dice; those are different tasks with different metrics. The
paper does no learned segmentation at all (teeth isolated via manual VIA
annotation). The real analog to this paper is the not-yet-built **module 6**
(shade classifier), not module 2.

Dataset differences: paper uses 500 patients, 16,640 images (balanced,
originally smaller, resampled to 1040/class), standardized capture (iPhone
13 Pro Max, fixed distance/zoom, 5 controlled clinical lights), ground truth
from a Vita Easyshade V spectrophotometer. Our NCT dataset is 782 train / 209
val generic tooth photos with no lighting protocol, no shade labels, and no
patient/clinical standardization — it exists for segmentation only.

Takeaways for module 6 design:
- They hit 97% with classical ML (RF/XGBoost) on flattened LAB pixel
  features from a small controlled ROI — evidence a simple classifier may be
  a strong baseline before reaching for ViT+CLIP.
- **CLAHE on the L-channel** to cancel lighting variation is a concrete
  technique we don't have yet and directly addresses real-world lighting
  variability in patient photos — worth adding to module 5's CIELab
  extraction.
- They crop to the **central 30%** of each tooth (edges/roots deemed
  unreliable for color) rather than using the full segmented region — worth
  considering instead of the full MedSAM mask when extracting tooth color.
- We currently have no shade-labeled dataset analogous to their
  spectrophotometer ground truth — this is an open gap for module 6, not
  just a scale difference.

## module_2b YOLO localization + threshold fix (2026-09-10)

Real inference output (15-image sample, `notebook-c.ipynb` variants) showed
0.5 vs 0.9 threshold behavior directly: 0.5 gives solid coverage but visible
gum/lip pink bleeding in; 0.9 hugs tooth boundaries much more tightly at the
cost of more fragmentation/holes. Fragmentation barely hurts module 5's
CIELab averaging (just fewer tooth pixels); gum contamination corrupts it
directly (pulls `a`/`L` off true enamel color). **Set `MASK_THRESHOLD = 0.9`
in `infer.py`** on this basis.

Root cause of the leakage identified as the generic centered box prompt
(`get_bbox_prompt`), not the MedSAM checkpoint itself — training used
tight, GT-derived boxes; inference was handing it a loose fixed-margin box
covering most of the image.

Considered fixing this with face-landmark-derived boxes (MediaPipe FaceMesh)
first, per the discussion docs. **Checked our own val images before
implementing and ruled it out**: 10 of 15 samples checked directly — none
contain a visible face (clinical/stock intraoral crops, cheek retractors, no
eyes/nose in any of them). A face landmarker needs a detectable face; it
would silently fail on this dataset. Landmarks remain a good idea for
module 4's full-face-selfie domain later, just not for this dataset now.

**Built instead**: `module_2b_yolo_localization/train_yolo.py` — trains a
YOLOv8n ("nano") detector directly on the NCT dataset's own existing
`labels/*.txt` YOLO bbox annotations (confirmed already present alongside
images/masks via module_0's exploration; no new labeling needed). Single
class (`0 = tooth`), Ultralytics' default images/labels folder convention
matches our dataset layout with no reformatting.

`module_3_medsam_inference/infer.py` updated: `get_bbox_prompt_yolo` uses
the trained detector's box (union of all detected boxes if more than one
fires, matching how training boxes spanned the *entire* GT mask — not
per-tooth), with a small expand-only margin (20px, matching module_2's
`BBOX_SHIFT` jitter scale). `USE_YOLO_PROMPT` toggle + automatic fallback to
the old generic box (whole-run if the YOLO checkpoint is missing, per-image
if YOLO finds nothing) keeps this a safe, reversible change rather than a
rewrite. `NUM_SAMPLE_IMAGES` bumped to 15 (from 3) to match the existing
threshold-comparison sample for a direct before/after.

`train_yolo.py` completed 50 epochs on Kaggle (final epoch: precision ~0.50,
recall ~0.79, mAP50 ~0.53, mAP50-95 ~0.24 — climbing steadily, not plateaued,
but this is a "roughly where's the tooth" detector so recall matters more
than precision here since `get_bbox_prompt_yolo` unions all boxes anyway).
`best.pt` published as Kaggle Dataset `hemantdarur/yolomodel-inference`
(`yolo_checkpoints/tooth_yolo/weights/best.pt`), wired into `infer.py`'s
`YOLO_CHECKPOINT_PATH`.

**Real 10-image comparison run** (`docs/notebook-c.ofyoloipynb.ipynb`,
stopped at 10/15 samples): 4/10 clean (tight masks, no visible gum bleed),
2/10 minor edge fringing, 3/10 genuinely bad — one noisy/fragmented mask with
real background leakage (`7018.jpg`), one heavy red/gum leakage on an
inflamed-gum image (`8764.jpg`), one near-total failure on an out-of-domain
stock ad photo with braces covering most of the tooth surface
(`IMG_2385.jpg`, arguably unfair to judge — the tooth itself is mostly
occluded by metal, not a localization or boundary problem).

**Conclusion**: the YOLO box fix worked (most images improved vs. the old
generic-box run), but it exposed a *second*, distinct failure mode on hard
cases: MedSAM's own mask boundary quality, not localization, is now the
bottleneck. This is the exact checkpoint condition set in
`docs/DISCUSSION_next_course_of_action.md`'s plan — "boundary-aware loss /
Light HQ-SAM swap, only if the YOLO-prompt fix still leaks." It still leaks
on ~30% of hard cases.

**Decision**: before committing to a full segmenter swap (real cost: new
checkpoint, new fine-tune run, new module), try the cheap fix first.
`cleanup_mask()` added to `infer.py` (morphological close + connected-
component filtering, keeps every component within `CLEANUP_MIN_COMPONENT_FRACTION`
of the largest one — not just the single largest, since a healthy anterior
row is naturally multiple components). This can only fix fragmentation-shaped
failures (like `7018.jpg`); it cannot fix color-leakage failures
(`8764.jpg`) or occlusion failures (`IMG_2385.jpg`) — those need an actual
boundary/architecture fix (Light HQ-SAM) if they persist.

**Ran** (`docs/notebook-c-newcomponetnt.ipynb`, full 15/15 images this time).
Result: `cleanup_mask` made **no visible difference** on any of the known-bad
cases — confirmed, not assumed. Full tally: 6/15 clean, 5/15 moderate edge
bleed, 4/15 genuinely bad, and the 4 bad cases turned out to be **three
different root causes**, none of them shape/noise problems cleanup could
ever fix:
1. `IMG_2385.jpg` — mask riddled with internal holes (one largely-connected
   blob, not separate blobs — nothing for connected-component filtering to
   drop).
2. `8764.jpg` — heavy red gum-color leakage on inflamed gums — a pixel
   classification problem, not a shape problem.
3. `y9.jpeg` — near-total failure on an out-of-domain stock ad photo with
   braces covering most of the tooth surface.
4. `y17.jpeg` (new in the full 15) — mask caught only the right half of the
   arch, missing the left half entirely — a YOLO **localization** miss on
   an atypical framing (smile-arc reference photo with overlay lines), not
   a segmenter problem.

**Decision, two parallel tracks** (both cheap, both done before committing
to the big architecture swap):

1. **Adaptive Interior Enamel ROI** (`get_interior_enamel_mask` in
   `module_3_medsam_inference/infer.py`, used by
   `module_5_integration/pipeline.py`'s `extract_cielab`) — per-connected-
   component `cv2.distanceTransform`, keeping only each tooth blob's
   interior core (`INTERIOR_ENAMEL_FRACTION = 0.5` of that blob's own max
   distance-to-edge). Directly protects the CIELab reading from boundary
   color contamination (case 2 above) without touching the segmenter. Zero
   training cost. Falls back to the full mask if erosion empties it out
   entirely. **Not yet run.**
2. **Light HQ-SAM zero-shot check** (`module_3c_hqsam_zeroshot/infer_hqsam.py`)
   — Rank #1's segmenter (`SysCV/sam-hq`, `vit_tiny`/"Light HQ-SAM"),
   completely unmodified pretrained weights, no fine-tuning, reusing the
   same trained YOLO detector for localization. Tests whether a model that
   has never seen our data already beats fine-tuned MedSAM on the boundary-
   quality failures (cases 1 and 2 above) before committing to the real
   cost of fine-tuning it. **Must run in a separate fresh Kaggle session**:
   `pip install segment-anything-hq` installs under the same
   `segment_anything` module name as the original `segment-anything`
   package (confirmed via PyPI) — the two conflict in one environment.
   Checkpoint (`sam_hq_vit_tiny.pth`, ~42.5MB) auto-downloads from
   `huggingface.co/lkeab/hq-sam` on first run (needs Kaggle internet
   enabled, already required for pip installs anyway).

   **First real run hit `KeyError: 'vit_tiny'`** — the actual "Light"
   TinyViT variant isn't registered in `sam_model_registry` from the pip
   package (needs separate setup code the package doesn't ship; see
   `github.com/SysCV/sam-hq/issues/130`, unresolved upstream). Switched
   `HQSAM_MODEL_TYPE` to `vit_b` (same backbone size as our MedSAM
   checkpoint — still a fair boundary-quality comparison, just not the
   smallest edge variant, which only matters later at the distillation
   stage).

   **Second real run hit `RuntimeError: Unexpected key(s) in state_dict`**
   (`mask_decoder.hf_token.weight` and other HQ-specific keys). Root cause:
   `pip install segment-anything-hq` pulls PyPI version 0.3, a stale
   snapshot whose `build_sam.py` still builds the plain (non-HQ)
   `MaskDecoder` — confirmed by diffing it against the current GitHub
   source, which builds `MaskDecoderHQ` (the class that actually has
   `hf_token`/`hf_mlp`/etc.) and matches the downloaded checkpoint. Fixed by
   installing straight from GitHub source
   (`pip install git+https://github.com/SysCV/sam-hq.git`) instead of PyPI.
   **Not yet run** with this fix — full reasoning in
   `module_3c_hqsam_zeroshot/notes.md`.

## DentalMedSAM-HQ Architecture Upgrade (2026-09-17)

Following the empirical analysis of `docs/notebooka818589fdd.ipynb` (where classical color suppression `module_1b` over-burned 50–99% of pixels and zero-shot HQ-SAM lacked dental domain awareness), the pipeline is upgraded to **DentalMedSAM-HQ** (`module_2_medsam_finetune/train_hq.py` and `module_3_medsam_inference/infer_hq.py`):

1. **Decoder Architecture**: Upgraded to `MaskDecoderHQ` from `sam-hq` (`SysCV/sam-hq.git`), adding High-Quality Mask Tokens (`hf_token` + `hf_mlp`) that fuse early ViT spatial features into the mask prediction.
2. **Boundary-Aware Loss**: $\mathcal{L}_{total} = \mathcal{L}_{Dice} + \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Boundary}$, using Euclidean distance transforms on ground-truth masks to heavily penalize misclassified pixels along the gingival margin.
3. **Gradient Accumulation**: `ACCUM_STEPS = 8` to simulate an effective batch size of 8 while respecting the SAM decoder's `BATCH_SIZE = 1` constraint.
4. **Lighting Normalization**: CLAHE applied to the CIELab `L` channel during preprocessing (`apply_clahe_l_channel`) to normalize lighting variations across clinical photos.
5. **Target Metric**: Validation DSC > **0.94** (up from the 0.8838 baseline).

## DentalMedSAM-HQ train_hq.py bug fixes (2026-09-17)

Two bugs found and fixed in `module_2_medsam_finetune/train_hq.py` before the first successful training run:

**Bug 1 — `ModuleNotFoundError: No module named 'monai'`**
- `train_hq.py` originally imported `monai.losses.DiceLoss`.
- MONAI is not pre-installed in standard Kaggle GPU containers (it's a medical-imaging add-on that the previous `train.py` always pip-installed explicitly).
- Fix: replaced with a native PyTorch `SoftDiceLoss` (`nn.Module` subclass), eliminating the MONAI dependency entirely. The formula is identical: `(2·intersection + ε) / (union + ε)` where intersection and union are computed over the soft probabilities (not thresholded), so gradients flow through normally.

**Bug 2 — `RuntimeError` in `criterion(probs, masks)` during training loop**
- `CombinedDentalLoss` was defined as an `nn.Module` with `__init__` (three sub-losses) but its `forward()` method was **never written** — the class definition ended at the closing brace of `__init__`.
- When `criterion(probs, masks)` was called in the training loop, `nn.Module.__call__` dispatched to `forward()`, which didn't exist, raising a `RuntimeError`.
- Fix: added the missing `forward(pred_probs, gt_masks)` method:
  ```python
  def forward(self, pred_probs, gt_masks):
      gt_float = gt_masks.float()
      return self.dice_loss(pred_probs, gt_float) \
           + self.bce_loss(pred_probs, gt_float) \
           + 0.5 * self.boundary_loss(pred_probs, gt_float)
  ```

## DentalMedSAM-HQ Training Results & Visual Evaluation (2026-09-18)

### 1. Training Outcome
- **Run**: 20 epochs completed on Kaggle T4 GPU (~89 minutes).
- **Validation DSC**: Peaked at **0.8628** (did not reach target of >0.94).
- **Final loss**: 0.2196.

### 2. Visual Inspection of Generated Crops (`docs/outputs_dental_hqsam`)
Inspection of the 15 validation image crops revealed that DentalMedSAM-HQ performed significantly **worse** than the previous YOLO + MedSAM baseline:
1. **Severe Gingival Bleed (`8764_tooth_crop.png`, `7018_tooth_crop.png`)**:
   - The entire upper maxillary gum ridge was falsely included as a thick pink band across the teeth.
   - The boundary loss and early ViT feature injection failed to distinguish pink inflamed gingiva from white enamel.
2. **Eaten Margins and Artifact Slivers (`IMG_2385_tooth_crop.png`, `7625_tooth_crop.png`)**:
   - Incisal and interproximal edges showed ragged bite marks and missing vertical cutouts.
   - Thin, noisy floating sliver artifacts appeared on lateral teeth.

### 3. Root Cause Analysis
- The HQ token fuses low-level spatial features from the frozen ViT backbone. Because MedSAM was pretrained on CT/MRI radiological scans (grayscale), its early spatial feature representations do not separate tooth enamel from gingiva.
- In contrast, the **YOLO-prompted MedSAM baseline** provided tight, tooth-specific bounding box prompts that naturally constrained the standard mask decoder from bleeding into the gingiva.

### 4. Decision & Archival to `trailnerror/`
- DentalMedSAM-HQ is formally marked as an unsuccessful experiment.
- The files were moved and renamed into `trailnerror/` for archival:
  - `module_2_medsam_finetune/train_hq.py` → `trailnerror/module2_train_hq.py`
  - `module_3_medsam_inference/infer_hq.py` → `trailnerror/module3_infer_hq.py`
- `module_5_integration/pipeline.py` was reverted to directly use the proven YOLO + standard MedSAM pipeline (`module_3_medsam_inference/infer.py`).
