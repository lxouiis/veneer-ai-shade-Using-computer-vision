# Doctor Review App

## What this is
A Streamlit web app so Doc Sans can review module_3's output (YOLO box +
fine-tuned MedSAM, frozen encoder) one image at a time: approve a predicted
tooth mask, or redraw it if it's wrong, then confirm whether the redraw
fixed it. This is a verification step on the *segmentation* pipeline, not
the shade classifier (module_6) - same scope as the existing CVD_Annotator
tool, but for teeth instead of veins.

## Why it reuses module_3's functions instead of reimplementing them
`app.py` imports `load_model`, `load_yolo_model`, `preprocess_image`,
`get_bbox_prompt_yolo`, `get_bbox_prompt`, and `run_inference` directly from
`module_3_medsam_inference/infer.py`. If this app had its own copy of that
logic, it would silently drift from whatever module_3 actually does on
Kaggle the first time either one gets edited - Doc Sans would end up
reviewing a different pipeline than the one in production.

## Which checkpoints
Frozen-encoder MedSAM (`notebook-a-meowmeow`, DSC 0.8838), and the
**original** YOLO detector (module_2b's `tooth_yolo`), not the
smile-finetuned one. The smile-finetune's own eval (`module_2d_eval_combined/
evaluate_new_yolo.py`) showed a near-doubled skip rate (12/238 vs 7/238) and
a lower median DSC despite fixing a few specific odd-angle cases - net wash
to mildly negative, not a clear win, so the original stays the default here.

## Why Google Drive, not local disk
This app runs as a hosted web link, not something Doc Sans installs. Even
though it's now deployed on an EC2 instance with real persistent disk (see
"Deploying" below - this changed from the original Streamlit Community Cloud
plan), Drive stays the backend regardless: it's what lets module_5's Kaggle
pipeline reach the same review data without you manually copying files
between your laptop, Kaggle, and the EC2 instance. The dataset lives in a
Drive folder, every review decision is written to a single `reviews.json` in
Drive, and approved/flagged images are uploaded there too. Same pattern
(OAuth refresh token, `drive.file` scope - sees only files it created, never
the rest of the Drive) as the existing CVD_Annotator tool - reused
deliberately rather than inventing a second approach to the same problem.

## One-time setup (do this before deploying)
1. **Google Cloud Console**: create/reuse a project, enable the Drive API,
   create an OAuth 2.0 Client ID (type "Desktop app"), download its JSON as
   `client_secret.json` next to `setup_drive.py`.
2. Run `python setup_drive.py` locally - opens a browser consent screen once,
   creates the `tooth_shade_review` root folder (with `images/`,
   `approved_masks/`, `needs_review/` subfolders) in your Drive, and prints
   four secrets: `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`,
   `GDRIVE_REFRESH_TOKEN`, `GDRIVE_ROOT_ID`.
3. Run `python upload_dataset_to_drive.py /path/to/local/combined/images` to
   batch-upload the dataset (NCT + SmileImages, or whichever set Doc Sans
   should review) into the Drive `images/` folder. Safe to re-run if
   interrupted - skips files already uploaded.
4. Download the two checkpoints (`notebook-a-meowmeow`'s
   `medsam_tooth_best.pth`, module_2b's original `tooth_yolo/weights/
   best.pt`) into a local `models/` folder next to `app.py`.

## Deploying (AWS EC2, not Streamlit Community Cloud)
Originally planned for Streamlit Community Cloud, but switched to a single
AWS EC2 instance once the project had $200 in AWS new-account signup credits
available (not a student-specific program - a separate student credit
application is pending, unrelated to what's actually being used here)
- Streamlit Cloud's free tier (~1GB RAM) was a real risk for loading MedSAM's
ViT-B encoder + YOLO together, while a t3.medium (4GB) removes that risk
entirely at effectively zero net cost given the credits. Full runbook is in
`deploy/README.md`: launch the instance, get a free DuckDNS domain, run the
app as a systemd service, front it with Caddy for automatic HTTPS (so an
iPad shows a real padlock, not a warning).

## The "one-tap start" problem, and the Lambda that solves it
Running the instance 24/7 burns credits for no reason between Doc Sans's
review sessions - but if the instance is **off**, there's nothing running to
show her a toggle button; Streamlit only exists once the instance is already
on. `deploy/lambda_starter/` is a small AWS Lambda function (with its own
permanent Function URL, independent of the EC2 instance's state) that calls
`ec2.start_instances()`/`stop_instances()`, plus a static page (hosted on
S3, not the EC2 instance, for the same always-available reason) that she
actually bookmarks. Tapping **Start** calls the Lambda, polls until the real
app is reachable, then redirects - she never needs to know an EC2 instance
is involved. See `deploy/README.md` section 9 for the exact setup steps.
The IAM policy for the Lambda's execution role is scoped to exactly one
instance for Start/Stop (least privilege) - `DescribeInstances` has to stay
on `Resource: "*"` since AWS doesn't support scoping that specific action to
one resource, not because broader access was intended.

Starting isn't the whole problem, though - without a way to stop it too,
she'd have no choice but to leave it running (and billing) until someone
else notices. `app.py`'s sidebar has a **"Done for now - stop instance"**
button (behind a confirm/cancel step, so a misclick mid-review doesn't cut
her off) that calls the same Lambda with `action=stop`. It only appears if
`LAMBDA_STARTER_URL` is set in `.env` - unset (e.g. for local testing with
no real EC2 instance involved) just hides the button rather than breaking
anything.

## The review flow (what Doc Sans actually does)
Redesigned after looking at a teammate's separate (vein ultrasound)
annotation tool for UI/interaction ideas - not its code, just the pattern:
side-by-side photo/output, real Previous/Next through the full image list,
and a box-prompt tool that reruns the model, not just a brush. Still our own
backend (module_3's functions, Drive storage) throughout.

1. A navigation bar at the top shows Previous/Next, the current position
   ("Image 12 / 147"), and a status icon for the current image (⬜ not
   reviewed, ✅ approved, 🚩 needs review). Previous/Next move through
   **every** image, not just unreviewed ones - she can revisit anything,
   same as flipping through a physical stack of photos. On first load, the
   app jumps to the first not-yet-reviewed image automatically.
2. Below that: the original photo next to the predicted tooth mask overlay,
   then three choices:
   - **✅ Looks good** - mask uploaded to `approved_masks/`, status
     `approved`, moves to the next image.
   - **📦 Redraw the box** - opens a canvas showing the model's actual
     1024x1024 working frame (not the full-res photo - this has to be the
     same coordinate space MedSAM/YOLO work in, not the original photo's
     resolution). She draws one box; "Rerun with this box" calls
     `run_inference` again with it as the prompt, replacing YOLO's box for
     this image only, and lands back on the reviewing screen with the new
     prediction - she can approve it, redraw again, or switch to hand
     correction from there.
   - **✏️ Fix by hand** - the original freehand brush-correction canvas,
     unchanged from before, pre-sized to a comfortable on-screen scale then
     rescaled back to the original photo's resolution on save.
3. After a hand correction, she's asked **"Is this correct now?"**
   - Yes -> the corrected mask is uploaded to `approved_masks/`, status
     `approved_after_edit`.
   - No -> the original photo is copied to `needs_review/` for the team to
     look into separately, no mask is saved for it.
5. Sidebar shows running totals (reviewed / total, approved, needs review).

## What this produces, and what's still manual
At the end, `approved_masks/` in Drive holds doctor-verified tooth masks for
every approved image - a trustworthy, human-checked subset you can use with
more confidence than raw model output.
`needs_review/` is a triage list, not a dead end - someone still needs to
look at those images and decide what to do (bad photo? bad localization?
genuinely hard case?) - this app doesn't make that call for you.

## Where this output actually gets used: module_5
`approved_masks/` isn't just a sanity check - `module_5_integration/
pipeline.py` now actively prefers it over a fresh model prediction when a
reviewed mask exists for an image (see that module's own notes.md). Sync the
masks down locally with `download_approved_masks.py`, publish them as a
Kaggle Dataset the same way the checkpoints are, and attach it to the
session running module_5. A human-checked tooth mask is a meaningfully more
trustworthy color-extraction source than raw MedSAM output - this matters
most if module_5's output ends up feeding `module_6_vit_clip`'s pseudo-label
training data, where garbage-in from a bad mask would otherwise get baked
directly into training labels.
