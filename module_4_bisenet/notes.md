# Module 4: BiSeNet Face Parsing Notes

## What this module does
BiSeNet (Bilateral Segmentation Network) is a pretrained face-parsing model:
given a face photo, it labels every pixel with which facial part it belongs
to (skin, eyes, nose, lips, hair, etc). We use it, unmodified (no
fine-tuning needed), to extract just the **skin** and **lip** regions of a
patient's face — those crops feed into the CIELab color extraction in
module_5, alongside the tooth crop from module_3.

## Face parsing vs. segmentation
MedSAM (modules 2-3) does binary segmentation: tooth vs. not-tooth, one class.
BiSeNet does **multi-class** segmentation: every pixel gets one of 19
possible labels at once (skin, eyebrows, eyes, nose, lips, hair, etc). Both
are "segmentation" in the general sense, but face parsing specifically means
segmenting a face into its semantic parts.

## The 19-class label map
This module uses the same class scheme BiSeNet was trained on
(CelebAMask-HQ): 0 background, 1 skin, 2/3 eyebrows, 4/5 eyes, 6 eyeglasses,
7/8 ears, 9 earring, 10 nose, 11 mouth (inner mouth/teeth region), 12 upper
lip, 13 lower lip, 14 neck, 15 necklace, 16 clothing, 17 hair, 18 hat. We
only care about two of these: skin (`SKIN_LABELS = [1]`) and lips
(`LIP_LABELS = [12, 13]`, both upper and lower lip combined).

## Why ImageNet normalization
BiSeNet's backbone was pretrained on ImageNet, a huge general photo dataset,
before being fine-tuned for face parsing. Every ImageNet-pretrained model
expects its input normalized the same specific way it was trained: subtract
the ImageNet mean `(0.485, 0.456, 0.406)` and divide by the ImageNet standard
deviation `(0.229, 0.224, 0.225)`, per RGB channel. Feeding it differently
normalized input (like our simple 0-1 scaling in module_1) would shift every
pixel value away from what the network's learned weights expect, degrading
predictions significantly.

## Why the parsing map gets resized back up
BiSeNet's output is a 512×512 label map (matching its fixed input size), but
photos come in whatever resolution they were taken at. We resize the label
map back to the original photo's resolution using nearest neighbor
interpolation, for the same reason as mask resizing elsewhere in this
project: it's a map of class IDs, and interpolating between IDs (like
bicubic would) would invent meaningless in-between labels.

## What the output is and where it goes next
Skin and lip crop PNGs (background removed, same masking technique as
module_3's tooth crop) and a parsing visualization. Module 5's integration
pipeline calls these same functions to get skin and lip crops, then extracts
their CIELab color values alongside the tooth crop's.
