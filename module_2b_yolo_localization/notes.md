# Module 2b: YOLO Tooth Localization Notes

## Why this module exists
Module 3's `infer.py` currently prompts MedSAM with a generic box: the whole
image minus a fixed 10% margin. Reviewing real inference output showed this
causes gum/lip tissue to leak into the predicted tooth mask, because the box
tells MedSAM "the tooth could be almost anywhere in this large region" — a
much looser prompt than the tight, mask-derived boxes MedSAM was trained on
(see module_2's notes). This module trains a small object detector to find
the actual tooth region per photo, giving `infer.py` a real, per-image box
instead of a fixed guess. See `docs/DISCUSSION_next_course_of_action.md` for
the full reasoning behind why this was prioritized over swapping MedSAM for
a different architecture.

## Why YOLO, and why not the smile-line/landmark approach first
The original idea (see the discussion doc) was to derive the box from facial
landmarks (e.g. MediaPipe FaceMesh finding the mouth). Checking our own
validation images ruled that out: the NCT dataset is clinical/stock intraoral
photos — cheek retractors, tight crops on just the mouth, no eyes or nose
visible in any sample checked. A face landmarker needs a detectable face to
work at all, so it would silently fail on most or all of this dataset.

YOLO doesn't have that requirement — it trains directly on these exact
intraoral crops using the dataset's own existing bounding-box labels (already
sitting in `train/labels/` and `val/labels/`, confirmed via module_0's
exploration; nothing new had to be annotated). Landmark-based localization
still makes sense later, but only once real full-face patient selfies are the
input (module 4/BiSeNet's assumed domain) — not for this dataset.

## Why YOLOv8n ("nano")
The nano variant is the smallest/fastest size in the YOLOv8 family — matches
the "YOLO-Nano" idea from the research documents, and keeps this cheap on a
Kaggle T4: a detector's job here is just "roughly where is the tooth," not
fine-grained accuracy, so a small model is enough and trains fast.

## Why the dataset needed no reformatting
Ultralytics' training code finds each image's label file by swapping the
`images` folder for a `labels` folder at the matching path (e.g.
`train/images/foo.jpg` -> `train/labels/foo.txt`). Our dataset was already
laid out exactly this way from the start (confirmed by module_0's dataset
exploration) — `train_yolo.py` only needs to write one small YAML file
pointing at the dataset root; no copying or restructuring of the actual
image/label files.

## Single class, but per-tooth boxes
The dataset's YOLO labels use one class id, `0 = tooth`, but (confirmed from
a real Kaggle training run's `Instances` counts - hundreds of instances per
16-image batch) label **individual teeth**, not one box per photo. We don't
need per-tooth detections for the SAM prompt, so `infer.py`'s
`get_bbox_prompt_yolo` takes the union of every box YOLO fires on an image,
producing exactly the kind of single box spanning the *entire* visible tooth
region that module_2 trained MedSAM against (derived from the entire
ground-truth mask, not per-tooth). The detector is still trained on the
per-tooth labels as-is — we don't need to preprocess/merge the label files,
just merge the *predictions* at inference time.

## What happens to multi-box predictions
Handled by `get_bbox_prompt_yolo` taking the union of all detected boxes
rather than only the highest-confidence one - see above. This was anticipated
and built in from the start, not a fix added after seeing per-tooth labels.

## Handoff to module 3
Same Notebook A -> Kaggle Dataset -> Notebook B pattern already used for the
MedSAM checkpoint: run `train_yolo.py` in its own Kaggle session, publish the
resulting `best.pt` as a Kaggle Dataset, then point `infer.py`'s
`YOLO_CHECKPOINT_PATH` at it. **Not yet run** — this and `infer.py`'s changes
are written but unexecuted; running both and comparing against the existing
0.5-vs-0.9 threshold sample is the next concrete step.
