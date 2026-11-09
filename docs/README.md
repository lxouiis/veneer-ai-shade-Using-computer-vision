# AI-Based Veneer Shade Selection

## Goal
Given a patient face photo, automatically predict the correct VITA dental shade
label (A1–D4) for veneer selection.

## Pipeline
Three stages, built incrementally as Modules 0–5:

1. **MedSAM** (fine-tuned on the NCT tooth dataset) segments the patient photo
   and produces a **tooth crop**.
2. **BiSeNet** (used out of the box, pretrained on CelebAMask-HQ) parses the
   same photo and produces a **skin crop** and a **lip crop**.
3. **CIELab color values** are extracted from all three crops (3 values each
   → 9 numbers total). A **ViT + CLIP** contrastive model (built later, not
   part of Modules 0–5) matches the tooth image plus these 9 numbers against
   VITA shade text descriptions and outputs a shade label such as `"A2"`.

```
patient photo
   ├─► MedSAM  ──► tooth crop ─┐
   ├─► BiSeNet ──► skin crop ──┼─► CIELab (9 numbers) ─► ViT+CLIP ─► shade label
   └─► BiSeNet ──► lip crop ───┘        (Module 5)         (later)
```

## Modules

| Module | What it does | Output |
|---|---|---|
| `module_0_environment` | Verifies the environment (library versions) and explores the NCT dataset (folder structure, sample images/masks, YOLO label format). | Sanity-check figures in `/kaggle/working/module_0_environment/`. |
| `module_1_preprocessing` | Converts raw dataset images/masks into resized, normalized `.npy` arrays ready for MedSAM. | `.npy` files in `/kaggle/working/processed/{train,val}/{images,masks}/`. |
| `module_2_medsam_finetune` | Fine-tunes MedSAM's mask decoder on the NCT tooth masks (encoder frozen, mixed precision — fits a Kaggle T4). | Best checkpoint in `/kaggle/working/checkpoints/medsam_tooth_best.pth`. |
| `module_3_medsam_inference` | Runs the fine-tuned MedSAM on new photos and extracts a background-removed tooth crop. | Tooth crop PNGs + result figures in `/kaggle/working/outputs/`. |
| `module_4_bisenet` | Runs pretrained BiSeNet face parsing and extracts skin and lip crops. | Skin/lip crop PNGs + parsing figures in `/kaggle/working/outputs/`. |
| `module_5_integration` | Chains modules 3 and 4, extracts CIELab color values from all three crops, and saves per-patient outputs. | `patient_id/{tooth,skin,lip}.png` + `meta.json` in `/kaggle/working/pipeline_outputs/`. |

Each module folder has its own `notes.md` explaining the concepts used in
that module in plain English.

## Libraries used, and why

- **cv2 (OpenCV)** — fast image I/O and color space conversion (BGR↔RGB,
  RGB→LAB).
- **numpy** — array math: normalization, binarization, masking, averaging.
- **skimage.transform** — resizing with explicit interpolation control
  (bicubic for photos, nearest neighbor for masks/label maps).
- **matplotlib** — visualizing samples, masks, overlays, and parsing results
  for sanity-checking every module.
- **torch / torch.nn / torch.nn.functional** — the deep learning framework
  used to fine-tune and run MedSAM and BiSeNet.
- **segment_anything** — the SAM/MedSAM model architecture (`sam_model_registry`).
- **monai.losses** — medical-imaging-focused loss functions, specifically
  `DiceLoss` for segmentation training.
- **torchvision.transforms / PIL** — standard preprocessing (resize,
  normalize) for BiSeNet's ImageNet-style input pipeline.

## Dataset

**NCT tooth dataset**, Kaggle path:
`/kaggle/input/datasets/alielhenidy/tooth-dataset/Tooth dataset v4/`

```
train/
  images/  — 782 .jpg/.jpeg files
  masks/   — 804 .png files, named like imagename_mask.png
  labels/  — YOLO .txt files (bounding boxes, not shade labels)
val/
  images/  — 209 .jpg/.jpeg files
  masks/   — 213 .png files, named like imagename_mask.png
  labels/  — YOLO .txt files
```

Confirmed by actually running module_1 on Kaggle (2026-09-03): every image
has a matching mask (0 unmatched images in either split), but the dataset
also ships extra orphan masks with no corresponding image - 22 in train, 4 in
val. So the real, usable pair count is **782 train / 209 val**, smaller than
a naive image-file or mask-file count would suggest. Every module that pairs
images with masks matches by filename and silently skips orphan masks, rather
than assuming a 1:1 count. (Label `.txt` counts haven't been independently
re-confirmed against this mounted copy the way image/mask counts have.)

## How to run each module on Kaggle

Two Kaggle notebooks, matching the project's pipeline map:

- **Notebook A** — modules 0, 1, 2 (environment check, preprocessing, MedSAM
  fine-tuning). GPU-heavy, slow.
- **Notebook B** — modules 3, 4, 5 (MedSAM inference, BiSeNet parsing, CIELab
  extraction / integration). Depends on Notebook A's fine-tuned checkpoint.

Recommended for both: a Kaggle notebook with a **T4 or P100 GPU**, the NCT
tooth dataset attached, and internet enabled (for `pip install`).

```bash
pip install monai segment-anything
```

### Notebook A — modules 0, 1, 2

1. **module_0** — `explore_dataset.py`. Confirms the data looks right.
2. **module_1** — `preprocess.py`. Takes a few minutes; produces
   `/kaggle/working/processed/`.
3. **module_2** — download the original SAM ViT-B checkpoint
   (`sam_vit_b_01ec64.pth`) to `work_dir/SAM/` (see MedSAM's repo for the
   download link), then run `train.py`. Trains for `NUM_EPOCHS` (default 100),
   validating every 5 epochs, saving the best checkpoint to
   `/kaggle/working/checkpoints/medsam_tooth_best.pth` by validation DSC.
   Re-running this cell (e.g. after "Save & Run All") automatically **skips
   training** and just loads the existing checkpoint instead of burning GPU
   hours on a full retrain — set `FORCE_RETRAIN = True` at the top of
   `train.py` if you actually want to retrain from scratch.

Once training finishes, **Save Version** on Notebook A. This turns its
`/kaggle/working` output into a Kaggle Dataset that Notebook B can attach.

### Handoff: Notebook A → Notebook B

In Notebook B, **Add Data → Notebook Output** and attach Notebook A's saved
output. Kaggle mounts it at `/kaggle/input/<notebook-a-slug>/`, preserving
the working-dir-relative path — so the checkpoint lands at
`/kaggle/input/<notebook-a-slug>/checkpoints/medsam_tooth_best.pth`. Update
`CHECKPOINT_PATH` at the top of `module_3_medsam_inference/infer.py` to match
the actual slug Kaggle assigns (it's currently a placeholder,
`notebook-a-output`).

### Notebook B — modules 3, 4, 5

Also attach: the NCT tooth dataset (module_3 samples from `val/images`), the
`ipythonx/celebamaskhq` Kaggle dataset (modules 4–5 need a full face — BiSeNet
has nothing to parse in the NCT tooth close-ups; CelebAMask-HQ is a good
smoke test since BiSeNet was itself trained on it — swap for real patient
photos later), and BiSeNet's pretrained weights.

4. **module_3** — `infer.py`. Runs MedSAM inference on 3 sample val images.
5. **module_4** — download BiSeNet's pretrained face-parsing weights
   (`79999_iter.pth`, from `zllrunning/face-parsing.PyTorch`) to
   `/kaggle/working/bisenet/`. `SAMPLE_IMAGES_DIR` currently points at
   `/kaggle/input/celebamaskhq/CelebA-HQ-img` — after attaching the dataset,
   run `!ls /kaggle/input/celebamaskhq/` and fix the path if the folder name
   differs, then run `parse_face.py`.
6. **module_5** — `pipeline.py`, to produce full per-patient outputs and
   color vectors.

**GPU budget note**: Kaggle allots ~30 GPU-hours/week with up to ~9–12h per
session. Because module_2 trains only the mask decoder (encoder frozen, mixed
precision), Notebook A should comfortably fit in a single session. Notebook B
is much lighter (inference only) and should run quickly.

## Expected outputs

- `module_1`: ~3.2GB of `.npy` files under `/kaggle/working/processed/`.
- `module_2`: `medsam_tooth_best.pth`, with printed val DSC/IoU each
  validation epoch (target: DSC above ~0.80 for a working fine-tune).
- `module_3`/`module_4`: PNG crops + comparison figures per sample image.
- `module_5`: one folder per patient under `/kaggle/working/pipeline_outputs/`,
  each with `tooth.png`, `skin.png`, `lip.png`, and `meta.json` (9-number
  color vector, bbox, mask pixel areas, and any flags such as
  `"lip_region_not_found"`).

## Author
Sanskriti Patil, JNMC Belgaum

## Technical support
Ruchita
