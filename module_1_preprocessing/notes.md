# Module 1: Preprocessing Notes

## What is preprocessing, and why do we do it?
Preprocessing is the step where we take raw data (in our case, photos of teeth
and their corresponding segmentation masks) and transform it into a consistent,
clean format that a neural network can actually learn from. Raw photos come in
different sizes, different pixel value ranges, and sometimes inconsistent file
formats. If we fed raw, unprocessed data straight into a model:
- Images of different sizes couldn't be batched together for training.
- Pixel values in the 0-255 range make training less numerically stable than
  smaller, normalized ranges.
- Masks might have stray non-binary values (e.g. from JPEG compression
  artifacts or anti-aliased edges) that confuse a segmentation model expecting
  clean 0/1 labels.

Preprocessing once, up front, and saving the result means every future training
run just loads ready-to-use arrays instead of repeating this work every time.

## What does normalization mean, and why the 0-1 range?
Normalization means rescaling pixel values from their raw range (0-255 for a
standard 8-bit image) down to a smaller, consistent range (0-1) by dividing by
255. We do this because:
- Neural networks train more stably when input values are small and centered,
  rather than large integers like 0-255.
- Many pretrained models (like MedSAM) were themselves trained on inputs
  normalized this way, so we must match that expectation for transfer learning
  to work correctly.
- It keeps gradients during backpropagation from becoming too large or too
  small early in training.

## What is a binary mask?
A binary mask is an array the same height and width as the image, where every
pixel is either 0 or 1 (instead of a full grayscale/color range). In our case:
- 1 means "this pixel belongs to a tooth."
- 0 means "this pixel is background (not a tooth)."

This is the standard format for segmentation ground truth: it tells the model
exactly which pixels it should learn to predict as foreground vs. background.
We binarize because the raw mask PNGs can have slightly non-pure values
(e.g. 254 instead of 255) at edges due to image compression or resizing, and we
need a strictly two-class target for training.

## What are .npy files, and why are they faster than .jpg?
.npy is NumPy's native binary file format for saving arrays. Compared to .jpg:
- .jpg is a compressed image format: every time you load it, the CPU has to
  decode/decompress the image and you lose exact pixel values (JPEG is lossy).
- .npy stores the raw array bytes directly (uncompressed, or simply structured),
  so loading it is just reading bytes into memory - no decoding step, and no
  precision loss.
- Since we already did the expensive work (resize, normalize) once, saving as
  .npy means every future training epoch loads the exact final array instantly,
  instead of redoing resize/normalize/decompress on every single load.

The tradeoff is that .npy files take more disk space than .jpg, but for
training speed, that tradeoff is worth it.

## What is bicubic interpolation?
Interpolation is how we fill in pixel values when resizing an image to a new
size. Bicubic interpolation estimates each new pixel by looking at a 4x4
neighborhood (16 surrounding pixels) in the original image and fitting a smooth
curve through them. This produces smoother, higher-quality results than simpler
methods like nearest neighbor, which is why we use it (`cv2.INTER_CUBIC`) for
resizing the actual photographic images.

We do NOT use bicubic for masks. Masks are label maps (each pixel is a class
id: 0 or 1), not photographic content. Interpolating between labels with a
smooth curve could invent fake in-between values (e.g. 0.5) that don't
correspond to any real class. So for masks we use nearest neighbor
interpolation (`cv2.INTER_NEAREST`) instead, which simply copies the nearest
original pixel's exact value, preserving clean class boundaries. We then
binarize the mask again afterward as a safety net.

(An earlier version of this script used `skimage.transform.resize` for this,
which supports the same interpolation orders but is dramatically slower in
practice - especially with `anti_aliasing=True`, which runs an extra
Python/SciPy Gaussian filtering pass before every resize. `cv2.resize` gives
the identical bicubic/nearest-neighbor result using OpenCV's much faster
native implementation, which matters a lot over ~1000 images.)

## Libraries used, and why each one
- **os**: builds file paths and lists directory contents in a cross-platform
  way, so the script works on any OS without hardcoded path separators.
- **cv2 (OpenCV)**: reads image and mask files from disk, converts color
  spaces, and resizes both (bicubic for images via `INTER_CUBIC`, nearest
  neighbor for masks via `INTER_NEAREST`) - all with OpenCV's fast native
  implementations.
- **numpy**: does the array math (normalization, binarization) and handles
  saving/loading the .npy files.
- **tqdm**: shows a live progress bar while processing ~1000 images, so a
  multi-minute run doesn't look frozen.
- **matplotlib.pyplot**: visualizes a processed sample (image, mask, overlay)
  at the end, so we can visually confirm the preprocessing worked correctly.
