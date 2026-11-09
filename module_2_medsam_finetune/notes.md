# Module 2: MedSAM Fine-Tuning Notes

## What this module does
MedSAM (a medical-imaging version of Meta's Segment Anything Model) is
pretrained on general medical images, not teeth specifically. Fine-tuning
means taking those pretrained weights and continuing to train the model, but
only on our NCT tooth masks, so it learns to segment teeth well specifically.
This module does that: loads MedSAM, trains it on the `.npy` pairs from
module_1, validates every few epochs, and saves the checkpoint with the best
validation score.

## What is fine-tuning / transfer learning?
Training a large model from scratch needs huge amounts of data and compute.
Transfer learning instead starts from a model that already learned general
visual features (edges, textures, shapes) from a much bigger dataset, and
adapts just enough of it to our specific task. Here, MedSAM already
understands "how to segment things given a box prompt" in general medical
images; fine-tuning teaches it "what a tooth looks like" specifically.

## Why we freeze the image encoder (and prompt encoder)
MedSAM has three parts: an **image encoder** (turns the photo into features),
a **prompt encoder** (turns the bounding box into features), and a **mask
decoder** (combines both to predict the mask).
- We freeze the **prompt encoder** because a plain bounding box doesn't need
  any tooth-specific learning — its job is the same regardless of the image
  domain.
- We freeze the **image encoder** mainly for memory: backpropagating through
  a full Vision Transformer image encoder at 1024×1024 needs far more GPU
  memory than a Kaggle T4 (16GB) has available. Freezing it means we don't
  store its gradients, and running it under `torch.no_grad()` also avoids
  storing its intermediate activations.
- Only the **mask decoder** (a small network) is actually trained. This still
  works well in practice — most of the useful "how do I turn a photo into
  features" knowledge transfers as-is; what needs adapting is mainly how
  those features get turned into a mask.

## Bounding box prompts, and why we jitter them
MedSAM needs a bounding box prompt telling it roughly where to look. During
training we compute this box directly from the ground-truth mask (the
tightest box containing all foreground pixels). We then randomly expand each
edge of the box by up to `BBOX_SHIFT` pixels. This matters because at
inference time we won't have a perfect ground-truth box — we'll use an
approximate one (see module_3). Jittering the training boxes teaches the
model to still segment correctly even when the box isn't drawn perfectly
tight, which is exactly the situation it will face at inference time.

## Why BATCH_SIZE must be 1
This one is subtle and worth understanding if you hit it again elsewhere.
SAM's `mask_decoder` was designed around a specific use case: **one image,
multiple prompts** (e.g. a user clicking on several different objects in the
same photo, and getting a mask back for each click). Internally, it expands
the single image's embedding to match however many prompts came in, using
`torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)`.

Our training setup is the opposite shape: **multiple different images, one
prompt each** (a normal training batch). The decoder has no way to tell these
two situations apart - it just sees a number ("how many prompts") and repeats
the image batch by that many again. With `BATCH_SIZE = 2`, it treats the 2 as
"2 prompts for 1 image" and repeats our already-2-image batch to 4, while the
prompt embeddings correctly stay at 2 - causing a shape mismatch (`4` vs `2`)
deep inside the decoder. Setting `BATCH_SIZE = 1` sidesteps this entirely,
since "1 prompt" doesn't change anything when repeated. This is also why most
MedSAM fine-tuning examples you'll find online process one image per forward
pass, rather than a performance choice - it's this decoder constraint.

## Data augmentation, and why we added it after the first run
Our first full training run plateaued around DSC 0.87-0.88, well below a
published benchmark figure of 0.97 for this kind of task. One likely
contributor: the first run had **zero data augmentation** - every epoch saw
the exact same 782 training images, unchanged. A model that only ever sees
one fixed version of each photo can struggle to generalize to slightly
different lighting or framing it hasn't seen, which shows up as a lower
validation score.

We added two augmentations, applied only to the training set (never
validation, since validation must measure real performance on unmodified
data):
- **Random horizontal flip**: mirrors the image left-right. The mask has to
  be flipped the exact same way, or the "ground truth" tooth region would no
  longer line up with the (now-mirrored) photo.
- **Random brightness/contrast jitter**: only touches the image, never the
  mask, since dimming or brightening a photo doesn't change which pixels are
  actually tooth vs. background.

Together these mean the model sees a slightly different version of each
photo every epoch, instead of memorizing 782 fixed images - which generally
improves how well it generalizes to genuinely new photos at inference time.

## Resuming training instead of starting over
`RESUME_FROM_BEST` lets a follow-up training run continue from the existing
best checkpoint (e.g. to try the effect of adding augmentation) instead of
either skipping training entirely or discarding all prior progress and
starting from the original base SAM weights again. Before resuming, we
re-validate the loaded checkpoint once to find its *real* current DSC and use
that as the starting `best_dsc` - otherwise, if an early epoch of the new run
happened to score worse (e.g. while the model is still adjusting to seeing
augmented images), the script could overwrite a genuinely better checkpoint
with a worse one.

## Unfreezing the encoder, and why it needs its own (smaller) learning rate
After the first run plateaued around DSC 0.87-0.88, we tried unfreezing the
image encoder (`FREEZE_IMAGE_ENCODER = False`) as a second experiment,
building on top of the augmentation change above. This was originally kept
frozen specifically to fit a T4's 16GB VRAM under an assumed `BATCH_SIZE=2`;
since `BATCH_SIZE` turned out to be locked to 1 anyway (see the SAM decoder
note above), that memory concern is smaller than originally planned for, so
it became worth trying - with real remaining OOM risk, not a guarantee.

Critically, the encoder does **not** use the same learning rate as the
decoder. The encoder already has good pretrained features (from SAM's
original large-scale training); the decoder we've been training from a much
more random starting point. Using one shared learning rate for both risks
**catastrophic forgetting** - a normal-sized learning rate applied to
already-good pretrained weights can overwrite/destroy those useful features
in just a few steps, rather than gently refining them. `ENCODER_LEARNING_RATE`
is set 10x smaller than the decoder's `LEARNING_RATE` for exactly this
reason - a standard technique when fine-tuning a pretrained backbone
alongside a newer, still-learning head.

## Mixed precision (AMP)
Mixed precision training runs most operations in 16-bit floating point
instead of 32-bit, which uses less GPU memory and runs faster on modern GPUs,
while keeping a 32-bit "master copy" of weights for stability. We use
`torch.amp.autocast` (picks the right precision automatically per operation)
together with a `GradScaler` (scales the loss up before backprop so small
gradients don't underflow to zero in 16-bit, then scales back down before the
optimizer step).

## Loss function: Dice + BCE
- **BCEWithLogitsLoss** (binary cross-entropy) checks, pixel by pixel, how
  confident and correct the model's tooth/not-tooth prediction is.
- **DiceLoss** instead looks at the *overlap* between predicted and true
  tooth regions as a whole, which handles class imbalance better (most
  pixels in a photo are background, not tooth).
- We add them together so the model is pushed toward getting both individual
  pixels right AND getting the overall tooth shape/overlap right.

## Metrics: DSC and IoU
- **DSC (Dice Similarity Coefficient)**: `2 × overlap / (predicted area + true area)`.
  Ranges 0 (no overlap) to 1 (perfect match).
- **IoU (Intersection over Union)**: `overlap / union`. Also 0 to 1, but
  generally a bit stricter/lower than DSC for the same prediction.
Both measure how well the predicted mask matches the ground truth; we use
validation DSC specifically to decide which checkpoint is "best."

## What the output is and where it goes next
The saved checkpoint (`medsam_tooth_best.pth`) is the fine-tuned mask
decoder's weights (combined with the frozen encoders). Module 3 loads this
checkpoint to run inference on new, unseen photos and extract tooth crops.
