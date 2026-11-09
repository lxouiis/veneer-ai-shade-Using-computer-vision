# Tooth Shade Pipeline — Raw Research, Findings, Ideas, Rankings, and Next Steps

> Raw working notes from the discussion so far.  
> Purpose: preserve the entire direction in one place without turning it into a polished paper.  
> Status: brainstorming + experiment history + paper takeaways + proposed research direction.  
> Important: some sections are **already observed in our Kaggle experiments**, some come from the **uploaded paper**, and some are **proposed ideas that still need experiments**.

---

# 0. Overall Goal

Build a tooth-shade prediction pipeline from a patient photo that:

- automatically identifies/segments the visible teeth
- avoids including gum/lip pixels that corrupt tooth color
- extracts reliable CIELab color information
- uses skin/lip context when useful
- predicts a VITA-style tooth shade
- works under uncontrolled real-world lighting
- avoids overfitting on a relatively small dental dataset
- is feasible to train on Kaggle, mainly a T4-class GPU
- stays lightweight enough that a future version could run on edge/mobile devices
- provides some kind of confidence / quality signal when segmentation is bad
- ideally contains a useful research contribution rather than only swapping one pretrained model for another

Current broad pipeline:

```text
Patient photo
    ↓
Tooth segmentation
    ↓
Tooth crop
    ↓
Skin + lip segmentation
    ↓
CIELab feature extraction
    ↓
Image + numeric/color/context features
    ↓
Shade classifier
    ↓
Shade label + confidence + ΔE
```

Original modular plan:

```text
Module 0 — environment + dataset exploration
Module 1 — preprocessing
Module 2 — MedSAM fine-tuning
Module 3 — MedSAM inference
Module 4 — BiSeNet face parsing
Module 5 — CIELab extraction
Module 6 — ViT / classifier
Module 7 — integration + validation
```

---

# 1. What the Reference Paper Actually Does

Paper discussed:

**“Tooth color prediction in intraoral images under different clinical lights using ML algorithms and CLAHE technique: an In-Vivo study”**  
Lasers in Medical Science, 2025.

## Core task in the paper

The paper is solving:

> Can a machine-learning system predict tooth color accurately even when clinical lighting changes?

The paper is mainly about **tooth-shade classification**, NOT learned tooth segmentation.

That distinction matters.

Their reported **97% is classification accuracy**, not segmentation Dice/DSC.

So:

```text
Paper's 97% accuracy != our 0.8838 segmentation DSC
```

They measure different tasks.

The paper manually annotates/isolate tooth regions using VIA annotations, then performs color-based classification.

---

# 2. Hardware Used in the Paper

The paper explicitly lists:

```text
Processor: Intel Core i7-13700H
RAM: 16 GB
Development: Anaconda
Python: 3.8.15
OpenCV: 4.11.0
Labeling: VIA
Image capture: iPhone 13 Pro Max
```

No GPU is listed in the hardware table.

This makes sense because they used classical ML models rather than training a large deep neural network.

Algorithms used:

- Random Forest
- XGBoost
- Decision Tree
- KNN
- Logistic Regression

Reported overall accuracies:

```text
Random Forest          97%
XGBoost                97%
Decision Tree          96%
KNN                    89%
Logistic Regression    89%
```

So the paper shows that a relatively modest CPU/RAM machine can train a useful tooth-shade classifier when the input ROI and color preprocessing are good.

---

# 3. Paper Dataset and Capture Setup

Paper dataset:

- 500 volunteers
- 250 female / 250 male
- age 18–40
- 12 anterior teeth considered
- iPhone 13 Pro Max
- fixed capture distance
- x3 optical zoom
- controlled capture settings
- five lighting conditions

Lighting:

1. Natural light, no flash
2. Natural light + flash
3. Yellow light — 2700 K
4. Daylight — 4600 K
5. Daylight — 6500 K

Ground truth tooth shade came from a **Vita Easyshade V spectrophotometer**.

This is much more controlled than our current segmentation dataset.

Our NCT segmentation data is approximately:

```text
Train images: 782
Validation images: 209
Train masks: 804 files, with orphan annotations
Val masks: 213 files, with orphan annotations
Actual matched image-mask pairs are based on filename.
```

The NCT data is for segmentation and does NOT provide the same clinical shade labels / spectrophotometer ground truth.

This is an important gap for our eventual shade classifier.

---

# 4. Important Technique from the Paper: LAB + CLAHE

Paper preprocessing:

```text
RGB image
    ↓
LAB color space
    ↓
CLAHE applied only to L channel
```

CIELab:

```text
L = lightness
a = green ↔ red
b = blue ↔ yellow
```

Why use LAB:

- better aligned with human perception of color difference than RGB
- useful for dental shade matching
- separates luminance from chromatic components

Why CLAHE on L:

- reduce variation caused by illumination
- correct brightness locally
- avoid directly modifying a/b color components

Potential pipeline idea:

```text
RGB
 ↓
LAB
 ↓
preserve a,b
 ↓
CLAHE on L
 ↓
extract robust tooth color statistics
```

But we should ABLATE:

```text
raw L,a,b
vs
CLAHE-L + a,b
vs
raw L + CLAHE-L + a,b
```

Because too much normalization of L could remove information that actually matters for shade.

---

# 5. Important Technique from the Paper: Central Tooth ROI

A particularly important observation:

The paper says edges and roots are less reliable for color because of lighting and other effects.

Instead of using every tooth pixel, they use approximately the **central 30% of the tooth** as the ROI for color analysis.

This matters a lot for our pipeline.

Our current Module 5 averages CIELab across all foreground pixels in the predicted tooth mask.

That means:

```text
one thin strip of gum
+
one bright glare region
+
tooth edges
```

can alter the average color.

So even a segmentation with high DSC can still be bad for shade prediction.

---

# 6. Proposed Improvement: Adaptive Interior Enamel ROI

Rather than copying a fixed rectangular central 30%, we proposed:

## Adaptive Interior Enamel ROI

Pipeline:

```text
predicted tooth mask
    ↓
distance transform
    ↓
calculate distance of each tooth pixel from mask boundary
    ↓
remove the outer boundary ring
    ↓
keep deepest / safest enamel pixels
    ↓
extract CIELab
```

Concept:

```text
Full tooth mask

████████████████
████████████████
████████████████

Interior safe region

   ██████████
   ██████████
   ██████████
```

Possible rule:

```text
keep pixels whose boundary distance >
k × maximum interior distance
```

Example k might be 0.20–0.35 and should be validated.

Why this may be stronger than fixed central 30%:

- respects actual tooth geometry
- works on differently sized teeth
- avoids gingival margin
- avoids tooth edges
- reduces specular highlight / boundary contamination
- almost zero inference cost
- highly edge-friendly
- directly optimized for the actual shade task

This may improve downstream shade accuracy even without dramatically improving segmentation DSC.

---

# 7. Current MedSAM Training Setup

Current MedSAM architecture has:

```text
Image Encoder
Prompt Encoder
Mask Decoder
```

Current strategy:

- image encoder frozen
- prompt encoder frozen
- mask decoder trained
- binary tooth segmentation
- Dice + BCE loss
- 1024×1024 MedSAM working resolution
- batch size effectively forced to 1 in the current training implementation

Reason for freezing encoder:

- Kaggle T4 has 16 GB VRAM
- full ViT backpropagation is much heavier
- pretrained visual features should already be useful
- small dataset increases overfitting risk if the entire encoder is trained

Current best checkpoint:

```text
Best validation DSC = 0.8838
```

Current checkpoint is the frozen-encoder run.

---

# 8. MedSAM Experiment History

## Run 1

Configuration:

- frozen image encoder
- frozen prompt encoder
- train mask decoder
- no augmentation
- 100 epochs
- batch size 1
- AMP enabled

Result:

```text
Best val DSC = 0.8838
```

This is the current working checkpoint.

## Run 2

Configuration:

- resumed from run 1
- image encoder unfrozen
- encoder learning rate = 1e-5
- decoder LR roughly 10× larger
- augmentation enabled
- 20 epochs

Validation trajectory:

```text
epoch 5   ~0.8727
epoch 10  ~0.8782
epoch 15  ~0.8701
epoch 20  ~0.8763
```

It never beat the 0.8838 baseline.

Training also became ~2.4× slower.

Conclusion:

> Blindly unfreezing the full encoder is not currently worth the Kaggle compute.

Could longer training eventually help? Possibly. But it is not the first thing to spend GPU time on.

---

# 9. Batch Size Constraint

Current SAM decoder handling caused a shape mismatch for batch size > 1.

SAM internally expects something closer to:

```text
1 image + N prompts
```

while normal training wants:

```text
N images + 1 prompt each
```

Batch size 2 caused tensor repetition / shape mismatch.

Current workaround:

```text
BATCH_SIZE = 1
```

Potential future workaround:

```text
gradient accumulation
```

But this is not currently the highest-priority problem.

---

# 10. Current Main Segmentation Failure

The biggest visible problem is not that MedSAM cannot see teeth.

It is:

> gum/lip leakage along the tooth boundary.

Current Module 3 uses a generic inference box:

```text
image
 ↓
box = roughly whole image minus 10% margin
 ↓
MedSAM
```

During training the bbox came from the ground-truth mask and was jittered.

During inference the model has NO ground truth, so the generic box tells it:

> “the object could be almost anywhere in this large region.”

That allows nearby gum/lip tissue to be included.

This creates a training/inference prompt mismatch.

---

# 11. Threshold Experiment: 0.5 vs 0.9

We compared the same 15 validation images using the same checkpoint and random seed.

## Threshold 0.5

Observed:

- solid tooth blobs
- good coverage
- more gum/lip leakage
- visible pink strip often included at gingival boundary

## Threshold 0.9

Observed:

- mask hugs tooth boundaries more tightly
- much less pink gum/lip included
- masks may fragment into vertical tooth strips
- more holes/gaps
- some disconnected pieces

For the shade pipeline:

```text
gum contamination = very harmful
small internal hole = less harmful
```

Why?

Because Module 5 averages CIELab values over predicted foreground pixels.

Gum pixels strongly change:

```text
a channel (green/red)
L channel (brightness)
```

while a hole simply removes some tooth pixels from the average.

Therefore:

```text
0.9 threshold currently looks better than 0.5 for shade extraction.
```

This is a cheap improvement with no retraining.

But it does NOT solve the model's uncertainty at the tooth/gum boundary.

---

# 12. Current Root Cause Hypothesis

Primary hypothesis:

```text
bad / overly broad prompt
    ↓
MedSAM sees teeth + surrounding tissue as plausible foreground
    ↓
gum leakage
```

Therefore:

> Before swapping architecture, fix prompt quality.

A sharper segmentation architecture receiving a poor box can still over-segment.

---

# 13. Smile Line / Smile Arc Idea

This came from an author/domain idea and may become useful.

Clinical concept:

> Smile line / smile arc relates the curve formed by the visible anterior teeth / incisal edges relative to the lips.

We discussed two main uses.

## Use A — Smile / mouth geometry as localization

Instead of:

```text
generic image box
```

use:

```text
face / lip landmarks
    ↓
mouth opening
    ↓
smile region / dental arc
    ↓
better SAM prompt or ROI
```

Possible implementation:

- MediaPipe FaceMesh or equivalent
- identify upper/lower lip landmarks
- derive mouth opening
- estimate where anterior teeth should appear
- create bbox / positive prompts from this region

Benefits:

- no manual user prompt
- no retraining needed at first
- handles full-face vs close-up photos better than a fixed image margin
- reduces search area for tooth segmentation

## Use B — Smile-Arc QA Metric

At deployment there is no GT mask.

Therefore you cannot calculate DSC for each incoming patient photo.

Proposed:

```text
predicted tooth mask
    ↓
extract relevant tooth boundary / incisal geometry
    ↓
fit smooth curve
    ↓
compare with plausible smile/dental arc
    ↓
ArcConsistencyScore
```

Example conceptual output:

```text
0.95 = plausible segmentation
0.70 = questionable
0.30 = likely segmentation failure
```

Possible use:

```text
meta.json:
{
  "segmentation_confidence": ...,
  "smile_arc_score": ...,
  "needs_review": ...
}
```

This is interesting because it becomes a **reference-free inference-time QA feature**.

Important:

Smile-line geometry should NOT become the sole segmentation method.

Better role:

- identifier/localizer
- prompt prior
- geometric constraint
- quality-control signal

---

# 14. Candidate Segmentation Models / Ideas Discussed

Models / concepts considered:

- MedSAM
- AutoSAMUS
- Tooth-ASAM
- 3DTeethSAM
- SegmentAnyTooth
- Light HQ-SAM
- YOLO11-Nano / small YOLO
- U-Net
- DeepLabV3+
- adapters
- learned Auto-Prompt Generator (APG)
- knowledge distillation
- boundary-aware loss
- lightweight mask refiner
- dental localization from landmarks
- adaptive enamel ROI

Important design principle:

> Do NOT stack five giant published systems together.

Novelty should come from solving a specific failure with a minimal coherent mechanism.

---

# 15. AutoSAMUS — What We Want to Borrow

Key idea discussed:

## Auto-Prompt Generator (APG)

Instead of manually supplying:

```text
bbox / points
```

the model learns to generate prompt embeddings from the image/features.

Conceptually:

```text
image
 ↓
encoder features
 ↓
small APG
 ↓
automatic prompt
 ↓
SAM decoder
```

Why relevant:

Our biggest current failure is prompt quality.

An APG could potentially eliminate the generic bbox entirely.

Why not immediately rank #1:

- implementation complexity
- training an APG is additional research work
- 782 images is not a huge dataset
- YOLO / mouth landmark localization is easier and more mature

Still very interesting for eventual novelty.

---

# 16. Tooth-ASAM — What We Want to Borrow

Main useful concept:

## Dental/domain adapters

Rather than unfreeze the entire pretrained image encoder:

```text
large frozen encoder
+
small trainable adapters
+
decoder
```

Possible adapter bottleneck:

```text
768
 ↓
64
 ↓
activation
 ↓
768
```

Exact dimensions depend on chosen backbone.

Benefits:

- fewer trainable parameters
- lower overfitting risk
- lower VRAM
- faster Kaggle training
- retains pretrained visual representation
- easier eventual distillation

For only ~782 training pairs, this is more sensible than full encoder fine-tuning.

---

# 17. SegmentAnyTooth — Why It Ranked Highly

Concept discussed:

```text
image
 ↓
YOLO-Nano
 ↓
tooth bounding box
 ↓
Light HQ-SAM
 ↓
precise tooth mask
```

Why it matches our problem:

- designed around tooth/intraoral localization + segmentation
- detector solves prompt/localization issue
- nano detector is edge-friendly
- Light HQ-SAM is much more deployment-oriented than giant MedSAM
- our dataset ALREADY contains YOLO-format bbox labels

Important point:

We are currently not using those YOLO annotations for segmentation.

That means we have cheap supervision available for:

```text
tooth detector
```

without creating a completely new annotation set.

---

# 18. 3DTeethSAM — What We Think

Not a direct fit.

Its main domain is 3D dental data / meshes, whereas our input is:

```text
RGB patient photograph
```

We can borrow concepts such as:

- lightweight refinement
- prompt generation
- dental domain adaptation
- mask refinement

But using the entire architecture would add complexity that is unrelated to our input modality.

Therefore it ranked low for direct implementation.

---

# 19. Lightweight U-Net / DeepLabV3+

Another option:

```text
image
 ↓
small supervised segmenter
 ↓
binary tooth mask
```

Advantages:

- no prompt
- simpler architecture
- easy Kaggle training
- easy edge deployment
- direct single-class tooth segmentation

Disadvantages:

- only ~782 train pairs
- potentially higher overfit risk from scratch
- may lose benefits of giant pretrained visual representation

Best role:

## baseline / student model

Could be extremely useful in the paper for comparison.

---

# 20. Proposed New Architecture Concept

Working concept name:

## DentalAutoSAM-Lite

Not necessarily final project/paper name.

Concept:

```text
                    Patient Photo
                         │
                         ▼
               Lightweight / frozen encoder
                         │
                 dental adapter blocks
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
    Auto prompt / detector       boundary features
            │                         │
            └────────────┬────────────┘
                         ▼
                lightweight SAM decoder
                         │
                         ▼
                    raw mask
                         │
                         ▼
                  mask refinement
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
        smile-arc QA        adaptive enamel ROI
                                     │
                                     ▼
                                CIELab stage
```

This would combine ideas rather than copy an individual model.

---

# 21. YOLO vs APG

Two possible ways to solve the prompt problem.

## Option A — YOLO Nano

```text
image
 ↓
tiny detector
 ↓
bbox
 ↓
SAM / Light HQ-SAM
```

Pros:

- mature
- easy to train
- existing YOLO labels
- fast
- edge friendly
- good Kaggle feasibility
- very interpretable

Cons:

- two-stage pipeline
- less novel than a learned prompt embedding
- detector failure propagates to segmentation

## Option B — Auto Prompt Generator

```text
image features
 ↓
small learned APG
 ↓
prompt embedding
 ↓
SAM decoder
```

Pros:

- fully automatic
- more elegant
- potentially more research novelty
- removes explicit bbox detector

Cons:

- more custom implementation
- more research risk
- smaller data may limit learning
- harder to debug

Current position:

> Use YOLO / landmarks first. If that solves the problem, then decide whether APG is worth building as the research contribution.

---

# 22. Boundary-Aware Loss

Current loss:

```text
Dice + BCE
```

Proposed:

```text
Ltotal =
    LDice
  + LBCE
  + λ LBoundary
```

Reason:

Our main segmentation error is at the **tooth-gum boundary**.

Global DSC may remain high even when a narrow band of gum is included.

For shade prediction:

```text
small boundary error can produce large color error
```

Therefore boundary quality deserves explicit optimization.

Potential alternatives:

- Boundary Loss
- Hausdorff-style loss
- Dice + Focal + boundary
- weighted BCE with extra weight near ground-truth boundaries

Need ablation experiments.

---

# 23. Downstream / Color-Aware Segmentation Loss

More novel proposed idea:

Instead of optimizing only mask overlap, optimize for the downstream task.

Concept:

```text
L =
Lsegmentation
+
λ1 Lboundary
+
λ2 Lcolor_contamination
```

Possible `Lcolor_contamination`:

- compare predicted-mask LAB statistics to GT tooth-mask LAB statistics
- penalize predicted foreground that strongly resembles gum/lip relative to GT
- penalize significant change in enamel color statistics caused by mask error

Important caution:

This must be designed carefully so the network does NOT learn shortcuts based on skin tone / shade distribution.

Potentially useful as an auxiliary loss, not primary supervision.

Research framing:

> downstream-aware segmentation for tooth-shade estimation

This is more meaningful than optimizing DSC only.

---

# 24. Robust Color Statistics

Current Module 5 primarily extracts mean CIELab values from:

```text
tooth
skin
lip
```

giving 9 numbers:

```text
L_tooth a_tooth b_tooth
L_skin  a_skin  b_skin
L_lip   a_lip   b_lip
```

Potential improvement:

Means are sensitive to:

- specular glare
- small gum contamination
- shadows
- outlier pixels

Possible robust alternatives:

```text
median
trimmed mean
MAD
standard deviation
10th percentile
90th percentile
interquartile range
```

Need to keep dimensionality controlled.

One candidate:

```text
tooth:
median L,a,b
trimmed mean L,a,b
dispersion L,a,b

skin:
median L,a,b

lip:
median L,a,b
```

Still lightweight.

---

# 25. Skin + Lip as Illumination Context

Currently skin and lip colors are planned as classifier inputs.

Alternative / additional use:

## Relative color calibration

Create:

```text
ΔL_tooth_skin
Δa_tooth_skin
Δb_tooth_skin

ΔL_tooth_lip
Δa_tooth_lip
Δb_tooth_lip
```

Why:

Absolute tooth colors can shift because of:

- camera exposure
- illumination
- white balance
- phone sensor processing

Relative differences to nearby facial regions may capture some common illumination shift.

This costs almost nothing computationally.

Needs experimental validation because skin and lip color vary strongly across people.

Should be used as context, not as an assumption that there is a fixed “normal” skin/lip color.

---

# 26. BiSeNet Current Role

Current Module 4:

- pretrained BiSeNet face parsing
- no fine-tuning
- extracts skin
- extracts upper + lower lips
- CelebAMask-HQ pretrained
- works on tested face samples

Why it exists:

```text
patient photo
 ↓
skin crop
lip crop
 ↓
CIELab context
```

Future edge concern:

MedSAM + BiSeNet + ViT + CLIP may become too heavy as an end-to-end mobile pipeline.

Eventually either:

- replace with lighter face parsing
- share one backbone between tasks
- run some components conditionally
- distill to a smaller student network

---

# 27. Knowledge Distillation

Long-term deployment idea:

Use a large accurate system as teacher.

```text
strong teacher:
YOLO + adapted SAM / Light HQ-SAM
    ↓
soft masks / pseudo labels / features
    ↓
small student
```

Possible student:

```text
MobileNetV3 encoder
+
small U-Net decoder
```

or another mobile segmentation network.

Deployment:

```text
patient image
 ↓
tiny student
 ↓
tooth mask
```

The large research model never needs to ship to the edge device.

This is one of the best ways to balance:

```text
high development accuracy
+
light deployment
```

---

# 28. Shared Encoder Long-Term Idea

Even more deployment-oriented:

```text
                   shared mobile encoder
                   /      |       \
                  /       |        \
             tooth      face      shade
             head       head      head
```

One backbone does shared feature extraction.

Tiny heads perform:

- tooth segmentation
- skin/lip parsing
- shade feature extraction/classification

Potential benefits:

- less memory
- lower latency
- easier edge deployment

But this is a future architecture, not the next experiment.

---

# 29. Overfitting Strategy

Major concern:

```text
~782 segmentation training images
```

We need to avoid memorization.

Recommended principles:

## Split first

```text
TRAIN
VALIDATION
TEST
```

Then augmentation ONLY on train.

If subject IDs exist:

```text
split by patient / subject
```

not random image-level split.

Why:

If multiple images from the same person enter both train and test, evaluation may become overly optimistic.

If patient IDs are unavailable:

- check filenames / metadata
- use perceptual hashing / embedding similarity to detect near duplicates
- keep near duplicates inside the same split

## Freeze large backbone initially

Train:

- adapters
- decoder
- APG / small detector
- lightweight heads

Avoid full ViT fine-tuning unless proven necessary.

## Other controls

- early stopping
- weight decay
- learning-rate scheduling
- small trainable parameter count
- realistic augmentation
- cross-validation when feasible
- untouched final test set
- monitor train vs val DSC/IoU
- monitor downstream color error as well

---

# 30. Augmentation Strategy

Segmentation deployment variation likely includes:

- brightness
- exposure
- white balance
- slight blur
- JPEG compression
- rotation
- scale
- translation
- mild perspective change
- shadows
- different framing

Potential segmentation augmentation:

```text
small rotation
small translate/scale
horizontal flip where anatomically acceptable
gamma/exposure changes
mild white-balance shifts
Gaussian blur
JPEG compression
mild shadow
```

Important warning:

Our downstream task is COLOR.

Do not use aggressive arbitrary hue/color jitter in a way that destroys real dental shade information.

Segmentation augmentation and shade-classifier augmentation should be designed separately.

---

# 31. Metrics We Should Care About

Do not use only DSC.

## Segmentation metrics

- Dice / DSC
- IoU
- Precision
- Recall
- boundary F-score / boundary IoU
- Hausdorff distance if useful
- gum contamination rate
- false-positive pixel area around gingival boundary

## Color-specific segmentation metrics

Possible:

```text
ΔE between GT-mask tooth color and predicted-mask tooth color
```

This is extremely relevant.

A model with:

```text
DSC 0.90
ΔE 1.2
```

may be better for our product than:

```text
DSC 0.94
ΔE 3.0
```

if shade accuracy is the goal.

## Deployment QA

- smile arc consistency
- segmentation confidence
- mask area ratio
- connected component count
- plausible location inside mouth
- enamel ROI size
- glare score
- blur score
- exposure score

## Final shade model

- shade classification accuracy
- macro F1
- per-class F1
- confusion matrix
- top-2 accuracy
- ΔE if continuous color ground truth exists
- calibration / confidence quality

---

# 32. Current Classification Plan

Original Module 6 concept:

```text
tooth image
+
9 CIELab numbers
+
VITA text embeddings
    ↓
ViT / CLIP / cross-attention
    ↓
shade prediction
```

Possible output:

```text
"A2"
confidence
ΔE
```

But the reference paper gives an important warning:

A simple classical ML baseline may be surprisingly strong.

Before building a complicated ViT+CLIP classifier, test:

- Random Forest
- XGBoost
- Logistic Regression
- small MLP

using robust LAB features.

This gives a baseline.

If a giant vision-language model only improves 0.5% while being 100× heavier, it may not be worth edge deployment.

---

# 33. Hardware / Kaggle Reality

Current Kaggle observations:

- P100 caused CUDA compatibility problems in the environment used
- T4 works
- Kaggle accelerator/session changes can reset installed packages
- MONAI / segment-anything may need reinstall
- full MedSAM training is expensive
- frozen encoder is much more practical
- current batch behavior is effectively 1
- storage was reduced significantly by saving uint8 arrays instead of float32

Preprocessing optimization already done:

```text
skimage resize ~23 minutes for 782 images
→ cv2.resize ~22 seconds
```

Storage improvement:

```text
~12.8 GB float32
→ ~3.2 GB uint8
```

These are important because Kaggle constraints should influence architecture decisions.

---

# 34. Top-10 Ranking Discussed

Ranking was based on:

- Kaggle / T4 feasibility
- expected training time
- accuracy potential
- overfitting resistance with ~782 images
- future edge deployment
- research novelty / usefulness
- implementation risk

## Rank 1 — YOLO-Nano → Light HQ-SAM / SegmentAnyTooth-style

Score discussed: ~9.3/10.

Why #1:

- solves prompt/localization issue
- uses lightweight detector
- uses lighter segmentation
- strong deployment path
- our dataset already has YOLO bbox labels
- feasible on Kaggle
- easy to run ablations
- low implementation risk relative to custom APG

Pipeline:

```text
image
 ↓
YOLO-Nano
 ↓
tooth bbox
 ↓
Light HQ-SAM
 ↓
mask
```

---

## Rank 2 — Frozen pretrained encoder + dental adapters + lightweight decoder

Score discussed: ~9.0/10.

Best for research/transfer-learning balance.

Why:

- few trainable parameters
- lower overfit risk
- preserves pretrained features
- Kaggle feasible
- can create a real dental adaptation contribution

---

## Rank 3 — Current MedSAM + learned Auto-Prompt Generator

Score discussed: ~8.7/10.

Why:

- directly attacks generic-box problem
- automatic
- potentially novel
- keeps most current model

Downside:

- custom implementation and research risk

---

## Rank 4 — YOLO / landmark bbox → current frozen MedSAM

Score discussed: ~8.6/10.

Why:

- cheapest serious improvement
- reuse 0.8838 checkpoint
- almost no MedSAM retraining
- lets us test whether prompt is the true bottleneck

This should happen BEFORE throwing away MedSAM.

---

## Rank 5 — Lightweight student via knowledge distillation

Score discussed: ~8.4/10.

Why:

- excellent eventual edge deployment
- can learn from stronger teacher
- useful after strong teacher exists

Not first because teacher must be improved first.

---

## Rank 6 — Boundary-aware MedSAM / adapter model

Score discussed: ~8.2/10.

Why:

- directly solves gum leakage
- shade pipeline is highly boundary-sensitive
- modest architectural change

---

## Rank 7 — Small U-Net / DeepLabV3+

Score discussed: ~7.8/10.

Why:

- very feasible
- lightweight
- simple
- good baseline

Concern:

- higher overfit risk from small dataset
- loses pretrained SAM-style representation unless pretrained backbone used

---

## Rank 8 — Full AutoSAMUS-style architecture

Score discussed: ~7.4/10.

Useful ideas:

- APG
- automatic prompting
- feature adaptation

But full reproduction is more complex than needed for our current stage.

---

## Rank 9 — Full Tooth-ASAM reproduction

Score discussed: ~7.1/10.

Main takeaway is adapters.

Better to borrow the dental adaptation idea than blindly reproduce every component.

---

## Rank 10 — 3DTeethSAM adaptation

Score discussed: ~4.8/10 for THIS task.

Not because the model is poor, but because input modality is wrong for us.

We have 2D RGB photos, not 3D dental meshes.

---

# 35. Separate Ranking: Smile-Line / Dental-Arc Guidance

Smile-line is not a model architecture, so it was ranked separately.

Conceptual score:

```text
~9.1/10 as a low-cost supporting technique
```

Strengths:

- almost no GPU training
- could improve localization
- could generate bbox prior
- could validate geometry
- provides QA without GT masks
- fits domain knowledge
- lightweight / edge friendly

Best use:

```text
supporting prior + QA
```

not sole segmentation.

---

# 36. Separate Ranking: Adaptive Interior Enamel ROI

Conceptual score for the OVERALL SHADE PIPELINE:

```text
~9.5/10
```

Reason:

It attacks the actual end task:

```text
accurate enamel color
```

rather than optimizing a visual mask metric only.

Very cheap computationally.

Potentially large effect on downstream shade accuracy.

---

# 37. Suggested Immediate Experiment Order

Because Kaggle GPU time is limited, do the cheap experiments first.

## Experiment 1 — Threshold

```text
current MedSAM
threshold 0.5
vs
threshold 0.9
```

Already visually studied.

0.9 currently preferred for shade extraction.

---

## Experiment 2 — Prompt Quality

Same exact MedSAM checkpoint.

Compare:

```text
A. generic bbox
B. YOLO bbox
C. mouth-landmark bbox
D. smile-line / dental-arc-assisted bbox
```

Measure:

- DSC
- IoU
- boundary metric
- gum contamination
- tooth LAB difference vs GT-mask LAB

This directly tests the root cause.

---

## Experiment 3 — Mask Cleanup

Compare:

```text
raw mask
vs
largest connected component
vs
morphological open/close
vs
small-hole fill
```

Be careful:

A multi-tooth mask may naturally contain multiple separate components.

Do NOT blindly keep only one largest connected component if the goal is to retain an entire anterior tooth row.

Better:

- remove only tiny isolated blobs
- connected-component area threshold
- preserve plausible tooth components
- morphology tuned to avoid merging teeth incorrectly

---

## Experiment 4 — Enamel ROI

Compare:

```text
full predicted mask LAB
central 30% ROI LAB
adaptive interior ROI LAB
```

Primary metrics:

- ΔE against GT-mask-derived safe enamel statistics if possible
- variance under lighting changes
- downstream shade accuracy when labels become available

---

## Experiment 5 — YOLO + Existing MedSAM

Train small tooth detector using existing YOLO annotations.

Then:

```text
YOLO bbox
 ↓
existing 0.8838 MedSAM
```

This is likely the highest-value short-term architecture experiment.

---

## Experiment 6 — Boundary Loss

Compare:

```text
Dice+BCE
vs
Dice+BCE+Boundary
```

Keep everything else identical.

---

## Experiment 7 — Dental Adapters

Frozen encoder:

```text
base frozen encoder
+
train small adapters
+
mask decoder
```

Compare trainable parameters, GPU memory, time, DSC, boundary quality.

---

## Experiment 8 — Light HQ-SAM

If YOLO localization is successful:

```text
YOLO
+
Light HQ-SAM
```

Compare against:

```text
YOLO
+
MedSAM
```

This tells us how much model size can be reduced without losing quality.

---

## Experiment 9 — Auto Prompt Generator

After detector baseline exists:

```text
Frozen encoder
+
APG
+
SAM decoder
```

Compare:

```text
APG vs YOLO prompt
```

Important research question:

> Can learned automatic prompting match/beat explicit tooth detection while remaining lightweight?

---

## Experiment 10 — Distillation

Only after teacher model is strong.

Teacher:

```text
best segmentation system
```

Student:

```text
mobile segmentation model
```

Optimize edge latency/model size.

---

# 38. Suggested End Architecture — Practical Version

Most practical near-term architecture:

```text
PATIENT PHOTO
     │
     ▼
YOLO-Nano tooth / mouth locator
     │
     ▼
existing MedSAM first
(or later Light HQ-SAM)
     │
     ▼
threshold / boundary cleanup
     │
     ├──────────────→ Smile-arc QA
     │
     ▼
Adaptive Interior Enamel ROI
     │
     ▼
RGB → CIELab
     │
     ▼
raw + CLAHE-L statistics
     │
     ├── skin/lip relative color context
     │
     ▼
Shade classifier
     │
     ▼
Shade + confidence + ΔE
```

This is the recommended first serious implementation.

---

# 39. Suggested End Architecture — Research Version

More novel architecture after baselines:

```text
PATIENT PHOTO
     │
     ▼
Frozen lightweight pretrained encoder
     │
Dental adapters
     │
     ├──────────────┐
     │              │
     ▼              ▼
Auto Prompt      Boundary feature
Generator        branch / refiner
     │              │
     └──────┬───────┘
            ▼
    Lightweight SAM decoder
            │
            ▼
       tooth mask
            │
     ┌──────┴───────────┐
     ▼                  ▼
Smile-arc QA     Adaptive Enamel ROI
                         │
                         ▼
                lighting-aware LAB
                         │
                         ▼
                shade classification
```

Possible contribution framing:

> **Downstream-aware automatic tooth segmentation for robust shade estimation under uncontrolled illumination.**

Potential contribution pieces:

1. automatic dental localization/prompting
2. parameter-efficient dental adaptation
3. boundary-aware tooth segmentation
4. geometry-aware enamel ROI
5. lighting normalization in Lab
6. smile/dental-arc plausibility QA
7. shade-oriented evaluation rather than segmentation-only evaluation
8. lightweight / distillable architecture

---

# 40. What NOT To Do

Avoid:

```text
MedSAM
+
Tooth-ASAM
+
AutoSAMUS
+
3DTeethSAM
+
SegmentAnyTooth
+
SAM2
+
multiple giant encoders
```

This would:

- use too much compute
- make Kaggle training painful
- make overfitting more likely
- make experiments impossible to interpret
- make ablation studies weak
- make edge deployment unrealistic
- make the “novel contribution” look like model stacking

Better:

> one clear failure → one clear mechanism → controlled ablation.

---

# 41. Important Research Question

The most important question is NOT:

> “How high can segmentation Dice go?”

It is:

> “How accurately can we recover the clinically useful tooth color / shade?”

Therefore segmentation should be evaluated downstream.

Example:

```text
Model A:
DSC = 0.94
but gum contamination causes ΔE = 3.2

Model B:
DSC = 0.90
but enamel ROI gives ΔE = 1.1
```

For this project, Model B may be superior.

That leads to a stronger research framing.

---

# 42. Potential Novel Metrics

Ideas to explore:

## Gum Contamination Ratio

```text
predicted foreground outside GT tooth
and near gingival region
/
predicted foreground
```

## Enamel Color Error

```text
ΔE(predicted-mask enamel stats, GT-mask enamel stats)
```

## Arc Consistency

```text
difference between predicted dental boundary geometry
and expected / detected smile arc
```

## Interior Enamel Stability

Take the same tooth / similar image under multiple lighting conditions and evaluate:

```text
variance of corrected LAB
```

A good pipeline should reduce lighting-induced spread.

## Edge Efficiency

Report:

```text
parameter count
model size MB
inference ms
peak memory
FLOPs / MACs where available
```

Then accuracy is not the only goal.

---

# 43. Edge Device Requirements

Eventually track:

- model parameter count
- serialized model size
- inference latency
- RAM usage
- peak GPU/accelerator memory
- quantization compatibility
- ONNX / TFLite / CoreML feasibility
- INT8 performance
- battery / thermal cost later

Candidate strategy:

```text
research teacher
→ distillation
→ quantized mobile student
```

Do not require the final phone app to run the same giant model used for research.

---

# 44. Potential Final Experimental Table

Eventually aim for something like:

| Model / Method | DSC | IoU | Boundary F1 | Gum Leakage | Tooth ΔE | Params | Latency | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Current MedSAM generic box | 0.8838 baseline | | | | | | | |
| MedSAM + 0.9 threshold | | | | | | same | same | |
| MedSAM + mouth bbox | | | | | | | | |
| MedSAM + YOLO bbox | | | | | | | | |
| MedSAM + boundary loss | | | | | | | | |
| MedSAM + dental adapters | | | | | | | | |
| YOLO + Light HQ-SAM | | | | | | | | |
| APG + lightweight SAM | | | | | | | | |
| tiny U-Net baseline | | | | | | | | |
| distilled student | | | | | | | | |

And separate color table:

| Color extraction | Shade Acc | Macro F1 | Mean ΔE | Lighting Stability |
|---|---:|---:|---:|---:|
| Full mask mean LAB | | | | |
| Central 30% LAB | | | | |
| Adaptive interior ROI | | | | |
| Adaptive ROI + CLAHE | | | | |
| Adaptive ROI + robust stats | | | | |
| + skin/lip relative features | | | | |

---

# 45. Key Risks

## Risk 1 — Segmentation dataset is not shade dataset

Our NCT dataset:

- good for segmentation
- not enough to train final VITA shade classifier unless shade labels exist elsewhere

Need shade-labeled data eventually.

## Risk 2 — Data leakage

Do not:

```text
augment
then split
```

Do:

```text
split
then augment train only
```

Prefer subject-level splitting.

## Risk 3 — Overfitting

Small data + giant ViT = danger.

Prefer:

- frozen encoder
- adapters
- small head
- validation monitoring

## Risk 4 — Optimizing wrong metric

High DSC does not guarantee accurate color.

Add shade-oriented metrics.

## Risk 5 — Color augmentation

Do not unintentionally destroy the signal we want to classify.

## Risk 6 — Model stacking

Keep architecture coherent and test every addition.

---

# 46. Current Recommended Decision

If choosing ONE practical direction right now:

## Do this first

```text
YOLO-Nano dental localization
        ↓
existing frozen MedSAM checkpoint
        ↓
MASK_THRESHOLD = 0.9
        ↓
small component / boundary cleanup
        ↓
smile-line / dental-arc QA
        ↓
adaptive interior enamel ROI
        ↓
CIELab
```

Why:

- reuses work already completed
- low Kaggle cost
- directly attacks observed failure
- uses labels already present
- gives measurable ablations
- gives useful answer about whether MedSAM itself is actually the bottleneck

Then:

```text
if this works:
    replace MedSAM gradually with lighter / dental-adapted architecture

if this still fails:
    investigate adapters / APG / Light HQ-SAM / small supervised model
```

---

# 47. Current Research-Focused Decision

If choosing ONE research architecture direction after the baseline tests:

```text
Frozen / lightweight encoder
+
dental adapter modules
+
automatic dental prompt generator
+
lightweight mask decoder
+
boundary-aware loss
+
smile-line QA
+
adaptive interior enamel ROI
```

This gives the strongest balance of:

- novelty
- Kaggle feasibility
- small dataset safety
- segmentation quality
- downstream relevance
- edge deployment potential

---

# 48. Short Version

What we currently believe:

```text
1. The paper's 97% is shade CLASSIFICATION, not segmentation.

2. The paper used i7-13700H + 16 GB RAM and classical ML.

3. Their strongest reusable ideas for us:
   - CIELab
   - CLAHE on L
   - avoid tooth boundaries
   - use a safe interior ROI
   - simple ML is a valid shade-classification baseline

4. Our MedSAM best:
   DSC = 0.8838.

5. Unfreezing full encoder:
   slower and did not beat baseline in the attempted run.

6. Current visible MedSAM problem:
   gum leakage.

7. Main suspected cause:
   generic inference bbox.

8. Threshold 0.9:
   cleaner gum boundary than 0.5,
   fragmentation is acceptable for color extraction.

9. First fix:
   better automatic localization/prompt.

10. Best practical candidate:
    YOLO-Nano + current MedSAM,
    later YOLO-Nano + Light HQ-SAM.

11. Best research ideas:
    dental adapters + APG + boundary loss.

12. Smile line:
    useful as mouth/tooth identifier, prompt prior, and QA score.

13. Very important new idea:
    Adaptive Interior Enamel ROI.

14. Very important metric shift:
    measure tooth color / ΔE, not DSC alone.

15. Edge plan:
    eventually distill strong teacher into a tiny student.
```

---

# 49. Immediate TODO

- [ ] Set / verify inference `MASK_THRESHOLD = 0.9`
- [ ] Re-run same fixed validation sample
- [ ] Train/use small YOLO tooth detector from existing labels
- [ ] Compare generic bbox vs YOLO bbox
- [ ] Try mouth landmark / smile-region bbox
- [ ] Implement boundary / tiny-component cleanup carefully
- [ ] Implement smile-arc QA score
- [ ] Implement adaptive interior enamel ROI
- [ ] Add raw LAB vs CLAHE-L extraction experiment
- [ ] Add robust color statistics
- [ ] Add tooth-vs-skin/lip relative ΔLab features
- [ ] Measure color error caused by segmentation
- [ ] Establish tiny U-Net/DeepLab baseline
- [ ] Test dental adapters only after prompt experiment
- [ ] Test Light HQ-SAM after localization baseline
- [ ] Build APG only after we know prompt quality is truly limiting
- [ ] Plan subject-level / leakage-safe train-val-test split
- [ ] Find / secure true shade-labeled dataset for Module 6
- [ ] Start final shade classifier with simple RF/XGBoost baseline
- [ ] Compare ViT/CLIP only after simple baseline exists
- [ ] Track model size / parameters / latency for edge goal
- [ ] Distill/quantize final winning model later

---

# 50. One-Line Research Direction

> Build an automatic, parameter-efficient, geometry-aware tooth segmentation and enamel-color extraction pipeline that is optimized not only for mask overlap but for accurate tooth-shade estimation under uncontrolled lighting, while remaining small enough to be distilled for future edge-device deployment.
