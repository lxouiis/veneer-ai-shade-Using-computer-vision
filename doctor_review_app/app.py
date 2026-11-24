import os
import sys
import json
import time
import tempfile
import threading
import urllib.request
from collections import OrderedDict
import numpy as np
import torch
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

from box_draw_component import box_draw_component
from drive_client import DriveClient

# module_3 is a sibling folder, not an installed package - import its real
# prediction functions by path so this reviewer always shows exactly what
# the production pipeline produces, not a second copy of the same logic
# that could silently drift from it.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJECT_ROOT, "module_3_medsam_inference"))

from infer import (  # noqa: E402
    load_model as load_medsam_model,
    load_yolo_model,
    preprocess_image,
    get_bbox_prompt_yolo,
    get_bbox_prompt,
    run_inference,
    run_inference_multi_box,
)

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# Checkpoints are the only things still local - small, fixed files that
# don't need to be shared/synced live the way the dataset and review state
# do. Download module_3's notebook-a-meowmeow checkpoint and module_2b's
# ORIGINAL tooth_yolo/weights/best.pt (not the smile-finetuned one - see
# notes.md for why) and point these at wherever they end up locally/in the
# hosting environment.
MEDSAM_CHECKPOINT_PATH = os.environ.get("MEDSAM_CHECKPOINT_PATH", "models/medsam_tooth_best.pth")
YOLO_CHECKPOINT_PATH = os.environ.get("YOLO_CHECKPOINT_PATH", "models/tooth_yolo_best.pt")

# The dataset, every review decision, approved masks, and flagged images all
# live in Google Drive instead of local disk - see setup_drive.py.
GDRIVE_ROOT_ID = os.environ["GDRIVE_ROOT_ID"]
REVIEWS_FILENAME = "reviews.json"

MASK_OVERLAY_COLOR = (255, 0, 0)  # red, drawn over the predicted tooth region
MASK_OVERLAY_ALPHA = 0.4

# Saved mask files encode 4 classes as distinct grayscale pixel values (not
# just a boolean) - this is what lets module_5_integration/pipeline.py tell
# a normal tooth pixel apart from a flagged one, instead of treating every
# painted pixel the same way. Values are spread out (not 0/1/2/3) so they
# stay visually distinguishable if anyone opens the raw PNG directly.
MASK_CLASS_VALUES = {"background": 0, "anomaly": 85, "ignore": 170, "tooth": 255}
# Canvas stroke colors per class - picked to be visually distinct from each
# other AND from typical tooth/gum photo colors, so decoding a drawn stroke
# back to a class by its RGB value (see decode_class_canvas) is unambiguous.
# "tooth" was originally pure white, which is nearly invisible when drawn
# on top of an actual (white) tooth - the whole point of drawing a stroke is
# to SEE what you've marked, so it needs real contrast against the thing
# it's labeling, not just uniqueness in RGB space.
CLASS_STROKE_COLORS = {
    "tooth": (0, 255, 0),          # bright green - matches CLASS_OVERLAY_COLORS
                                    # below, and reads clearly against white
                                    # enamel, pink gum, and skin tones alike
    "ignore": (255, 165, 0),       # orange
    "anomaly": (255, 0, 255),      # magenta
    "erase": (0, 0, 0),            # pure black - a dedicated "mark as
                                    # background" stroke, not just "unpainted."
                                    # Needed so erasing a mistake made earlier
                                    # in the same zoomed-in pass is actually
                                    # possible - without a distinct erase
                                    # color, there'd be no way to tell "never
                                    # painted here" apart from "deliberately
                                    # painted background," see decode below.
}
# For displaying each class as a translucent color over the photo - same
# idea as MASK_OVERLAY_COLOR/ALPHA above, extended to 3 distinguishable
# classes instead of one.
CLASS_OVERLAY_COLORS = {
    "tooth": (0, 255, 0),       # green
    "ignore": (255, 165, 0),    # orange
    "anomaly": (255, 0, 255),   # magenta
}

BRUSH_SIZE = 12  # px, for the hand-correction canvas
CANVAS_MAX_DIM = 900  # photos (and the 1024x1024 working frame) can be too
                       # large to draw on comfortably at full size; canvases
                       # are shown scaled down and converted back afterward

STATUS_ICON = {"not_reviewed": "⬜", "approved": "✅", "needs_review": "\U0001f6a9"}

# Same 16 shades module_6_vit_clip trains against (VITA Classical guide) -
# kept here too rather than imported, since this app and that training
# script run in completely separate environments (this one's a long-running
# server process, that one's a one-off Kaggle training job).
VITA_SHADES = [
    "A1", "A2", "A3", "A3.5", "A4",
    "B1", "B2", "B3", "B4",
    "C1", "C2", "C3", "C4",
    "D2", "D3", "D4",
]
NOT_SURE = "-- not sure / skip --"

# Optional - lets the sidebar show a "Stop Instance" button, so Doc Sans can
# end her own session instead of leaving the EC2 instance running (and
# billing) until someone else notices and stops it manually. Unset means the
# button just doesn't appear - the app still works fine without it, e.g. for
# local testing where there's no EC2 instance to stop at all.
LAMBDA_STARTER_URL = os.environ.get("LAMBDA_STARTER_URL")

# --------------------------------------------------------------------------
# PREDICTION CACHING (two layers, so revisiting a photo - via Previous/Next,
# or a whole new session days later - never re-pays the Drive download +
# full YOLO/MedSAM CPU inference for a photo already processed before).
#
# Layer 1 (session, in RAM): keyed by filename, holds the actual numpy
# arrays/tensor so a Previous/Next bounce within THIS browser session is
# instant - no Drive call, no inference, just a dict read. Capped so a long
# session reviewing hundreds of photos can't grow this unbounded; oldest
# entry is dropped first (classic LRU), since what matters is "the last N
# photos she was looking at," not every photo ever opened this session.
#
# Layer 2 (disk, on the EC2 box): keyed by filename, holds the downloaded
# image bytes + computed mask as plain files under CACHE_DIR. Survives a
# server restart/redeploy and a brand new session days later - the gap
# Layer 1 alone leaves open. On a disk-cache hit we still re-run
# preprocess_image (cheap resize/tensor-conversion) on the cached bytes
# rather than caching the tensor itself, so box-redraw still has a correct
# tensor to work with - only the Drive download and the expensive
# YOLO+MedSAM inference are actually skipped.
# --------------------------------------------------------------------------
SESSION_CACHE_MAX = 30
CACHE_DIR = os.environ.get("PREDICTION_CACHE_DIR", "/tmp/tooth_review_cache")

# --------------------------------------------------------------------------
# PREFETCH (compute the NEXT photo while she's still looking at/drawing on
# the current one, so clicking "Next" lands on an already-finished result
# instead of waiting for a fresh download + inference every time).
#
# Module-level, not st.session_state - a background thread can't safely
# touch Streamlit's session state (it belongs to the main script-run
# thread), so prefetched results land here first and get copied into the
# normal session cache the next time the main thread actually looks for
# them.
#
# _shared_resource_lock serializes every real call into the shared Drive
# client and the shared MedSAM/YOLO model objects - whether it's this
# background prefetch or a foreground "Rerun with this box" click. Without
# it, a prefetch thread and a foreground click could end up inside the same
# model/HTTP objects at the same time, which they're not written to handle
# safely. Whichever one asks first just goes first; the other waits its
# turn instead of racing - in practice a short wait, since she typically
# spends far longer looking at/drawing on a photo than one inference call
# takes.
_shared_resource_lock = threading.Lock()
_prefetch_cache = {}          # filename -> (image_tensor, image_resized_uint8, original_image, mask)
_prefetch_state_lock = threading.Lock()  # protects the two dicts below (cheap, not the heavy lock above)
_prefetch_in_progress = {}    # filename -> True, while a thread is working on it


def _take_from_prefetch_cache(filename):
    with _prefetch_state_lock:
        return _prefetch_cache.pop(filename, None)


def start_prefetch(filename, file_id, client, medsam_model, yolo_model):
    """Fire-and-forget: if `filename` isn't already cached/being fetched,
    kick off a background thread to download + predict it, one neighbor
    ahead - see run_inference_multi_box's sibling reasoning in infer.py for
    why a SINGLE shared encoder/connection matters, same idea here but
    across threads instead of across boxes."""
    with _prefetch_state_lock:
        if filename in _prefetch_cache or filename in _prefetch_in_progress:
            return
        _prefetch_in_progress[filename] = True

    def worker():
        try:
            with _shared_resource_lock:
                disk_image_bytes, disk_mask = load_from_disk_cache(filename)
                image_bytes = disk_image_bytes if disk_image_bytes is not None else client.download_bytes(file_id)
                suffix = os.path.splitext(filename)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name
                try:
                    image_tensor, original_image, image_resized_uint8 = preprocess_image(tmp_path)
                finally:
                    os.remove(tmp_path)
                if disk_mask is not None:
                    mask = disk_mask
                else:
                    bbox = None
                    if yolo_model is not None:
                        bbox = get_bbox_prompt_yolo(image_resized_uint8, yolo_model)
                    if bbox is None:
                        bbox = get_bbox_prompt(image_tensor.shape[-2:])
                    mask = run_inference(medsam_model, image_tensor, bbox)
                    save_to_disk_cache(filename, image_bytes, mask)
            with _prefetch_state_lock:
                _prefetch_cache[filename] = (image_tensor, image_resized_uint8, original_image, mask)
        except Exception:
            # Best-effort only - if this fails for any reason (a flaky
            # Drive call, anything), she just gets the normal slow path
            # when she actually navigates there. A background prefetch
            # failing silently is fine; it must never crash the app.
            pass
        finally:
            with _prefetch_state_lock:
                _prefetch_in_progress.pop(filename, None)

    threading.Thread(target=worker, daemon=True).start()


def _session_cache():
    if "prediction_cache" not in st.session_state:
        st.session_state.prediction_cache = OrderedDict()
    return st.session_state.prediction_cache


def get_from_session_cache(filename):
    cache = _session_cache()
    if filename not in cache:
        return None
    cache.move_to_end(filename)  # mark as most-recently-used
    return cache[filename]


def put_in_session_cache(filename, data):
    cache = _session_cache()
    cache[filename] = data
    cache.move_to_end(filename)
    while len(cache) > SESSION_CACHE_MAX:
        cache.popitem(last=False)  # drop the least-recently-used entry


def _disk_cache_paths(filename):
    safe_name = filename.replace("/", "_")
    image_path = os.path.join(CACHE_DIR, "images", safe_name)
    mask_path = os.path.join(CACHE_DIR, "masks", safe_name + ".npy")
    return image_path, mask_path


def load_from_disk_cache(filename):
    image_path, mask_path = _disk_cache_paths(filename)
    if not os.path.exists(image_path):
        return None, None
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    mask = np.load(mask_path) if os.path.exists(mask_path) else None
    return image_bytes, mask


def save_to_disk_cache(filename, image_bytes, mask):
    image_path, mask_path = _disk_cache_paths(filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    if not os.path.exists(image_path):
        with open(image_path, "wb") as f:
            f.write(image_bytes)
    np.save(mask_path, mask)


# --------------------------------------------------------------------------
# DRIVE SETUP (cached - folder IDs and the drive.file client don't change
# within a session, and re-resolving them on every rerun would mean extra
# API calls on every button click)
# --------------------------------------------------------------------------
@st.cache_resource
def get_drive():
    client = DriveClient.from_env()
    images_folder_id = client.get_or_create_folder("images", GDRIVE_ROOT_ID)
    approved_folder_id = client.get_or_create_folder("approved_masks", GDRIVE_ROOT_ID)
    needs_review_folder_id = client.get_or_create_folder("needs_review", GDRIVE_ROOT_ID)
    return client, images_folder_id, approved_folder_id, needs_review_folder_id


# --------------------------------------------------------------------------
# REVIEWS DB (reviews.json, stored as a single file in Drive - a few hundred
# entries is small enough to read/write whole, no need for a real database)
# --------------------------------------------------------------------------
def load_reviews_db(client):
    existing = client.find(REVIEWS_FILENAME, GDRIVE_ROOT_ID)
    if existing is None:
        return {}, None
    data = client.download_bytes(existing["id"])
    return json.loads(data), existing["id"]


def save_review(client, filename, status, mask_file_id=None, doctor_shade=None, corrected_boxes=None):
    db, reviews_file_id = load_reviews_db(client)
    db[filename] = {
        "status": status,  # "approved" | "approved_after_edit" | "needs_review"
        "mask_file_id": mask_file_id,
        # Dr. Sanskriti's own real shade judgement for this photo - separate
        # from, and stronger than, module_6_shade_classification's ΔE guess.
        # None means she didn't pick one (left on "not sure / skip") -
        # distinct from an actual answer, not treated as a guess either way.
        "doctor_shade": doctor_shade,
        # Her hand-drawn box(es) for this photo, in the same 1024x1024
        # working-frame coordinates the model itself uses - None if she
        # never redrew a box (approved YOLO's own guess as-is). This is
        # real, human-verified box ground truth for this photo - exactly
        # what a future YOLO fine-tuning pass needs, instead of only ever
        # training on YOLO's own (possibly wrong) predictions.
        "corrected_boxes": corrected_boxes,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload = json.dumps(db, indent=2).encode("utf-8")
    client.upload_bytes(REVIEWS_FILENAME, GDRIVE_ROOT_ID, payload, "application/json", file_id=reviews_file_id)
    return db


def status_of(filename, reviews_db):
    entry = reviews_db.get(filename)
    if entry is None:
        return "not_reviewed"
    return "approved" if entry["status"] in ("approved", "approved_after_edit") else "needs_review"


# --------------------------------------------------------------------------
# IMAGE LIST (built once per session, not re-fetched on every rerun - lets
# Previous/Next move through the FULL set, not just "next unreviewed," so
# she can revisit anything, same as flipping through a real stack of photos)
# --------------------------------------------------------------------------
def get_image_list(client, images_folder_id):
    if "image_list" not in st.session_state:
        entries = sorted(
            (f for f in client.list_children(images_folder_id)
             if f["name"].lower().endswith((".jpg", ".jpeg", ".png"))),
            key=lambda f: f["name"],
        )
        st.session_state.image_list = entries
        st.session_state.current_index = 0
    return st.session_state.image_list


def jump_to_first_unreviewed(reviews_db):
    for i, entry in enumerate(st.session_state.image_list):
        if entry["name"] not in reviews_db:
            st.session_state.current_index = i
            return
    st.session_state.current_index = 0  # everything reviewed - just start from the top


# --------------------------------------------------------------------------
# MODEL LOADING (cached across reruns - Streamlit reruns the whole script on
# every button click, so without caching this would reload both models,
# including a full ViT-B forward encoder, on every single interaction)
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    medsam_model = load_medsam_model(MEDSAM_CHECKPOINT_PATH)
    yolo_model = load_yolo_model(YOLO_CHECKPOINT_PATH)
    return medsam_model, yolo_model


# --------------------------------------------------------------------------
# PREDICTION (thin wrapper around module_3's real functions)
# --------------------------------------------------------------------------
def predict_mask(image_bytes, filename, medsam_model, yolo_model):
    # module_3's preprocess_image expects a real file path (cv2.imread), not
    # bytes - write the Drive download to a temp file first rather than
    # duplicating its decode logic here.
    suffix = os.path.splitext(filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        # preprocess_image returns the original full-resolution RGB image
        # (for display/overlay) separately from the 1024x1024 tensor/uint8
        # frame the model actually works in.
        image_tensor, original_image, image_resized_uint8 = preprocess_image(tmp_path)
    finally:
        os.remove(tmp_path)

    bbox = None
    if yolo_model is not None:
        bbox = get_bbox_prompt_yolo(image_resized_uint8, yolo_model)
    if bbox is None:
        bbox = get_bbox_prompt(image_tensor.shape[-2:])

    mask = run_inference(medsam_model, image_tensor, bbox)
    return image_tensor, image_resized_uint8, original_image, mask


def rerun_with_box(medsam_model, image_tensor, box):
    return run_inference(medsam_model, image_tensor, box)


def rerun_with_boxes(medsam_model, image_tensor, boxes):
    # One shared image_encoder pass for however many boxes she drew, not one
    # pass per box - see run_inference_multi_box's docstring in infer.py.
    return run_inference_multi_box(medsam_model, image_tensor, boxes)


def overlay_mask(image, mask):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)
    color_layer[mask_resized > 0] = MASK_OVERLAY_COLOR
    blended = overlay * (1 - MASK_OVERLAY_ALPHA) + color_layer * MASK_OVERLAY_ALPHA
    return blended.astype(np.uint8)


def overlay_class_mask(image, class_mask):
    """Same idea as overlay_mask, but class_mask holds one of
    "tooth"/"ignore"/"anomaly"/background (0) per pixel, each shown as its
    own translucent color - lets her see all 3 painted classes at once
    before saving, not just a single tooth/not-tooth view."""
    class_resized = cv2.resize(class_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)
    for class_name, value in MASK_CLASS_VALUES.items():
        if class_name == "background":
            continue
        color_layer[class_resized == value] = CLASS_OVERLAY_COLORS[class_name]
    present = class_resized != MASK_CLASS_VALUES["background"]
    blended = overlay.copy()
    blended[present] = (overlay[present] * (1 - MASK_OVERLAY_ALPHA)
                        + color_layer[present] * MASK_OVERLAY_ALPHA)
    return blended.astype(np.uint8)


def decode_class_canvas(image_data):
    """Fabric.js's canvas snapshot (RGBA at canvas resolution) -> (class
    values, touched) for this one drawing pass. class_values holds the
    saved-mask pixel value (0/85/170/255) wherever a known stroke color was
    drawn; touched marks every pixel actually painted THIS pass (tooth,
    ignore, anomaly, OR erase) so the caller can tell "genuinely erased on
    purpose" apart from "never touched in this pass" - both decode to the
    same background value otherwise, which would make erasing impossible to
    distinguish from doing nothing."""
    h, w = image_data.shape[:2]
    class_values = np.zeros((h, w), dtype=np.uint8)
    touched = np.zeros((h, w), dtype=bool)
    alpha = image_data[:, :, 3]
    rgb = image_data[:, :, :3]
    for class_name, stroke_color in CLASS_STROKE_COLORS.items():
        # A small tolerance, not an exact match - anti-aliased stroke edges
        # land on colors slightly off from the pure stroke_color value.
        close = np.all(np.abs(rgb.astype(np.int16) - np.array(stroke_color)) < 30, axis=-1)
        hit = close & (alpha > 0)
        class_values[hit] = MASK_CLASS_VALUES.get(class_name, MASK_CLASS_VALUES["background"])
        touched |= hit
    return class_values, touched


def view_window(h, w, zoom, cx, cy):
    """The visible (x0, y0, x1, y1) crop in full-image pixels for a given
    zoom level and center fraction - same approach as Wasif's DVT annotator
    (mask_editor.py's view_window), adapted for our single-image case. zoom
    1.0 = whole photo; higher values crop a smaller, more magnified region,
    which is what actually gives fine control - the brush stroke width stays
    the same in canvas pixels, but covers far fewer real photo pixels once
    zoomed in."""
    z = min(max(float(zoom), 1.0), 8.0)
    cw, ch = max(16, int(round(w / z))), max(16, int(round(h / z)))
    cw, ch = min(cw, w), min(ch, h)
    x0 = int(round(min(max(cx * w - cw / 2, 0), w - cw)))
    y0 = int(round(min(max(cy * h - ch / 2, 0), h - ch)))
    return x0, y0, x0 + cw, y0 + ch


def scale_for_canvas(h, w, max_dim=CANVAS_MAX_DIM):
    scale = min(1.0, max_dim / max(h, w))
    return int(h * scale), int(w * scale), scale


def rect_from_canvas(canvas_result):
    """Last drawn rectangle, in the canvas's own pixel coordinates - fabric.js
    (what st_canvas uses under the hood) applies width/height and a separate
    scaleX/scaleY on resize, so both have to be multiplied in, not just one."""
    if canvas_result is None or not canvas_result.json_data:
        return None
    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return None
    obj = objects[-1]
    x0, y0 = obj["left"], obj["top"]
    w = obj["width"] * obj.get("scaleX", 1)
    h = obj["height"] * obj.get("scaleY", 1)
    return x0, y0, x0 + w, y0 + h


# --------------------------------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Tooth Mask Review", layout="wide")
    # streamlit_drawable_canvas (fabric.js underneath) is a mouse-first,
    # "best effort" library with no documented touch/stylus support - on a
    # touchscreen, the browser's own default swipe-to-scroll/pinch-to-zoom
    # gesture handling intercepts taps before the canvas's own drawing code
    # ever sees them. touch-action: none tells the browser "don't treat
    # touches on this element as a scroll/zoom gesture" - a common real fix
    # for exactly this symptom, not guaranteed, but costs nothing to try.
    st.markdown(
        "<style>canvas { touch-action: none !important; }</style>",
        unsafe_allow_html=True,
    )
    st.title("Tooth Segmentation Review")
    st.caption("YOLO box + fine-tuned MedSAM (frozen encoder) - the same "
               "pipeline module_3 uses on Kaggle.")

    medsam_model, yolo_model = load_models()
    client, images_folder_id, approved_folder_id, needs_review_folder_id = get_drive()
    image_list = get_image_list(client, images_folder_id)
    # Loaded once per browser session, not re-fetched from Drive on every
    # single click - Streamlit reruns this whole function on every
    # interaction (moving a slider, switching the class radio, etc.), and
    # most of those reruns have nothing to do with reviews.json at all.
    # Updated in place after an actual save via save_review()'s return
    # value, not by reloading from Drive again.
    if "reviews_db" not in st.session_state:
        st.session_state.reviews_db, _ = load_reviews_db(client)
    reviews_db = st.session_state.reviews_db

    if not image_list:
        st.error("No images found in Drive's 'images' folder - run "
                 "upload_dataset_to_drive.py first.")
        return

    if "stage" not in st.session_state:
        jump_to_first_unreviewed(reviews_db)
        st.session_state.stage = "predicting"

    image_entry = image_list[st.session_state.current_index]
    filename = image_entry["name"]

    if "last_prediction_filename" not in st.session_state or \
            st.session_state.last_prediction_filename != filename:
        cached = get_from_session_cache(filename)
        if cached is None:
            # A background prefetch thread (started the last time she
            # looked at the PREVIOUS photo) may have already finished this
            # one while she was still reading/drawing - check that before
            # falling back to the slow disk/Drive path.
            cached = _take_from_prefetch_cache(filename)

        if cached is not None:
            # Layer 1 hit (RAM from earlier this session) or a finished
            # prefetch - either way, no Drive call, no inference needed now.
            image_tensor, image_resized_uint8, original_image, mask = cached
        else:
            with _shared_resource_lock:  # same lock the prefetch thread uses - see start_prefetch
                disk_image_bytes, disk_mask = load_from_disk_cache(filename)
                if disk_image_bytes is not None:
                    # Layer 2 hit - downloaded/computed in an earlier session.
                    # Drive download is skipped; preprocess_image is still run
                    # (cheap resize/tensor-conversion) so a later box-redraw has
                    # a real tensor to work with, but inference itself is
                    # skipped whenever the mask was cached too.
                    image_bytes = disk_image_bytes
                    suffix = os.path.splitext(filename)[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(image_bytes)
                        tmp_path = tmp.name
                    try:
                        image_tensor, original_image, image_resized_uint8 = preprocess_image(tmp_path)
                    finally:
                        os.remove(tmp_path)
                    if disk_mask is not None:
                        mask = disk_mask
                    else:
                        bbox = None
                        if yolo_model is not None:
                            bbox = get_bbox_prompt_yolo(image_resized_uint8, yolo_model)
                        if bbox is None:
                            bbox = get_bbox_prompt(image_tensor.shape[-2:])
                        mask = run_inference(medsam_model, image_tensor, bbox)
                        save_to_disk_cache(filename, image_bytes, mask)
                else:
                    # Full miss - never seen this photo before in any cache.
                    image_bytes = client.download_bytes(image_entry["id"])
                    image_tensor, image_resized_uint8, original_image, mask = predict_mask(
                        image_bytes, filename, medsam_model, yolo_model
                    )
                    save_to_disk_cache(filename, image_bytes, mask)
        put_in_session_cache(filename, (image_tensor, image_resized_uint8, original_image, mask))

        # Kick off the NEXT photo's download+prediction in the background
        # now, while she's about to spend time looking at/drawing on THIS
        # one - a no-op if it's already cached or already being fetched.
        next_index = st.session_state.current_index + 1
        if next_index < len(image_list):
            next_entry = image_list[next_index]
            start_prefetch(next_entry["name"], next_entry["id"], client, medsam_model, yolo_model)

        st.session_state.image_tensor = image_tensor
        st.session_state.image_resized_uint8 = image_resized_uint8
        st.session_state.original_image = original_image
        st.session_state.predicted_mask = mask
        st.session_state.last_prediction_filename = filename
        # A fresh photo means no box correction has happened for THIS photo
        # yet - without resetting these, approving a photo she never
        # redrew a box on could incorrectly inherit box data left over from
        # whatever photo she corrected previously in the same session.
        st.session_state.last_boxes_used = None
        st.session_state.per_box_masks_result = None
        st.session_state.stage = "reviewing"

    original_image = st.session_state.original_image
    mask = st.session_state.predicted_mask
    current_status = status_of(filename, reviews_db)

    # ---- Navigation bar ----
    nav_prev, nav_pos, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("← Previous", use_container_width=True,
                     disabled=st.session_state.current_index == 0):
            _go_to(st.session_state.current_index - 1)
    with nav_pos:
        st.markdown(
            f"<div style='text-align:center'>{STATUS_ICON[current_status]} "
            f"<b>Image {st.session_state.current_index + 1} / {len(image_list)}</b> "
            f"&mdash; {filename}</div>",
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("Next →", use_container_width=True,
                     disabled=st.session_state.current_index == len(image_list) - 1):
            _go_to(st.session_state.current_index + 1)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original photo", use_container_width=True)
    with col2:
        st.image(overlay_mask(original_image, mask), caption="What the model sees", use_container_width=True)

    # ---- Shade judgement (your own, not the model's) ----
    # Shown throughout, not just on the review buttons, so it stays visible
    # (and her pick stays selected) even if she goes through a box-redraw or
    # hand-edit first. Pre-filled from a prior session's answer, if there is
    # one - revisiting an already-reviewed photo via Previous/Next shouldn't
    # make her re-decide something she already answered.
    existing_entry = reviews_db.get(filename)
    prior_shade = existing_entry.get("doctor_shade") if existing_entry else None
    shade_options = [NOT_SURE] + VITA_SHADES
    default_index = shade_options.index(prior_shade) if prior_shade in VITA_SHADES else 0
    shade_choice = st.selectbox(
        "What VITA shade do you think this actually is? (your own judgement, not the model's guess)",
        shade_options, index=default_index, key=f"shade_{filename}",
    )
    doctor_shade = None if shade_choice == NOT_SURE else shade_choice

    # ---- Stage: reviewing ----
    if st.session_state.stage == "reviewing":
        approve_col, box_col, hand_col = st.columns(3)
        if approve_col.button("✅ Looks good", type="primary", use_container_width=True):
            mask_bytes = cv2.imencode(".png", (mask * 255).astype(np.uint8))[1].tobytes()
            mask_name = filename.rsplit(".", 1)[0] + "_mask.png"
            file_id = client.upload_bytes(mask_name, approved_folder_id, mask_bytes, "image/png")
            st.session_state.reviews_db = save_review(
                client, filename, "approved", file_id, doctor_shade,
                corrected_boxes=st.session_state.get("last_boxes_used"))
            _advance_after_review()

        if box_col.button("\U0001f4e6 Redraw the box", use_container_width=True,
                          help="Draw a box around the tooth yourself and have the model try again with it."):
            st.session_state.stage = "box_prompt"
            st.rerun()

        if hand_col.button("✏️ Fix by hand", use_container_width=True,
                           help="Draw directly on the mask with a brush - zoom in for "
                                "fine detail, add teeth the model missed entirely, or "
                                "flag an overlap/anomaly region."):
            st.session_state.stage = "editing"
            st.rerun()

    # ---- Stage: box_prompt (doctor draws a box, model reruns with it) ----
    elif st.session_state.stage == "box_prompt":
        # Drag-based drawing (the canvas below) has no documented touch
        # support and confirmed doesn't work reliably with touch/Apple
        # Pencil in practice - neither touch-action CSS nor a smaller
        # canvas fixed it. Tap mode sidesteps drag entirely: it's built on
        # a plain `onclick` handler (confirmed by reading the library's own
        # frontend source), which browsers reliably synthesize from a tap
        # on any touchscreen - a fundamentally different, more reliable
        # mechanism, not another guess on the same broken one.
        input_mode = st.radio(
            "How do you want to mark the box:",
            ["✏️ Draw directly (new - try this on touch/Pencil)",
             "🖱️ Drag a rectangle (mouse / trackpad)", "👆 Tap two corners (touch / Apple Pencil)",
             "🎚️ Adjust with sliders (most reliable on touch)"],
            horizontal=True, key=f"box_mode_{filename}",
        )

        small_canvas = st.checkbox("Smaller canvas (easier for touch / Apple Pencil)",
                                   key="box_small_canvas")
        max_dim = 420 if small_canvas else CANVAS_MAX_DIM

        canvas_h, canvas_w, scale = scale_for_canvas(
            *st.session_state.image_resized_uint8.shape[:2], max_dim=max_dim)
        display_frame = cv2.resize(st.session_state.image_resized_uint8, (canvas_w, canvas_h),
                                   interpolation=cv2.INTER_AREA)

        # A running list of boxes for THIS photo, in full working-frame
        # coordinates (not the scaled-down display ones) - lets her mark
        # several separate teeth with their own boxes instead of being
        # limited to one box per photo. Reset whenever she moves to a
        # different image, same as the other per-filename state above.
        if st.session_state.get("box_list_filename") != filename:
            st.session_state.box_list = []
            st.session_state.box_list_filename = filename

        if st.session_state.box_list:
            preview = display_frame.copy()
            for b in st.session_state.box_list:
                bx = [int(round(c * scale)) for c in b]
                cv2.rectangle(preview, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
            st.image(preview, caption=f"{len(st.session_state.box_list)} box(es) added so far",
                     use_container_width=True)

        if input_mode.startswith("✏️"):
            # Our own component, built from scratch on the Pointer Events
            # API (pointerdown/move/up) - a standard that unifies mouse,
            # touch, and pen input into ONE event model, unlike
            # streamlit-drawable-canvas's mouse-only sync-back (confirmed
            # broken on touch: box_prompt.py notes.md and today's testing -
            # the drawn shape renders but never reaches Python). This is
            # the real, direct fix rather than a workaround.
            st.info("Draw a box directly on the photo with your finger, "
                    "pencil, or mouse - then tap Rerun.")
            result = box_draw_component(display_frame, key=f"box_draw_{filename}")
            ready_box = [result["x1"], result["y1"], result["x2"], result["y2"]] if result else None
        elif input_mode.startswith("🎚️"):
            # Native Streamlit sliders - standard HTML range inputs, not a
            # third-party custom JS component. Guaranteed touch support on
            # every mobile browser, unlike the canvas (drag sync to Python
            # confirmed broken on touch) or the tap-coordinates component
            # (second tap confirmed not registering). Less elegant than
            # drawing directly, but doesn't depend on any touch-event
            # handling we don't control.
            st.info("Drag each slider with your finger to position the box, "
                    "then tap Rerun.")
            h_disp, w_disp = canvas_h, canvas_w
            left_col, top_col = st.columns(2)
            x0 = left_col.slider("Left edge", 0, w_disp, int(w_disp * 0.1), key=f"slider_x0_{filename}")
            y0 = top_col.slider("Top edge", 0, h_disp, int(h_disp * 0.1), key=f"slider_y0_{filename}")
            right_col, bottom_col = st.columns(2)
            x1 = right_col.slider("Right edge", 0, w_disp, int(w_disp * 0.9), key=f"slider_x1_{filename}")
            y1 = bottom_col.slider("Bottom edge", 0, h_disp, int(h_disp * 0.9), key=f"slider_y1_{filename}")

            preview = display_frame.copy()
            cv2.rectangle(preview, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), (255, 215, 0), 3)
            st.image(preview, use_container_width=True)

            ready_box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        elif input_mode.startswith("🖱️"):
            st.info("Draw a box around the tooth - click and drag one rectangle, "
                    "then tap Rerun.")
            canvas_result = st_canvas(
                fill_color="rgba(255, 215, 0, 0.15)",
                stroke_color="#ffd700",
                stroke_width=2,
                background_image=Image.fromarray(display_frame),
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="rect",
                key=f"box_canvas_{filename}",
            )
            ready_box = rect_from_canvas(canvas_result)
        else:
            # Two taps, no dragging: first tap sets one corner, second tap
            # sets the opposite corner. unix_time is compared against the
            # last-processed tap so the SAME click isn't re-applied on every
            # later rerun (the component keeps returning its last value).
            if st.session_state.get("tap_box_filename") != filename:
                st.session_state.tap_corner1 = None
                st.session_state.tap_box_filename = filename
                st.session_state.tap_last_time = None

            if st.session_state.tap_corner1 is None:
                st.info("Tap one corner of the box around the tooth.")
            else:
                cx, cy = st.session_state.tap_corner1
                st.info(f"Corner 1 set at ({cx}, {cy}) - now tap the opposite corner.")
                # Deliberately NOT redrawing a marker onto the image here -
                # that would change the image's encoded bytes, forcing the
                # <img> element to reload between the two taps. If the
                # second real tap lands while that reload/rebind is
                # happening, it can get swallowed - exactly the "first tap
                # works, second tap doesn't" symptom seen in testing. Same
                # unchanged image across both taps avoids that reload
                # entirely; the corner's coordinates are shown as plain
                # text above instead of as a visual marker on the photo.

            coords = streamlit_image_coordinates(display_frame, key=f"tap_{filename}")
            if coords and coords.get("unix_time") != st.session_state.tap_last_time:
                st.session_state.tap_last_time = coords["unix_time"]
                if st.session_state.tap_corner1 is None:
                    st.session_state.tap_corner1 = (coords["x"], coords["y"])
                    st.rerun()
                else:
                    x0, y0 = st.session_state.tap_corner1
                    x1, y1 = coords["x"], coords["y"]
                    st.session_state.tap_box_ready = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                    st.rerun()

            if st.button("Reset taps", use_container_width=True):
                st.session_state.tap_corner1 = None
                st.session_state.tap_box_ready = None
                st.rerun()

            ready_box = st.session_state.get("tap_box_ready")

        # Canvas/tap coords were shown scaled down from the 1024x1024
        # working frame - scale back up to that same space, which is what
        # the model actually expects, before storing/using any box.
        pending_box_full = [c / scale for c in ready_box] if ready_box is not None else None

        add_col, clear_col = st.columns(2)
        if add_col.button("➕ Add this box (for another tooth)", use_container_width=True,
                          help="Covers a different tooth/region with its own box. Leave "
                               "unused for the normal single-box case - just draw one box "
                               "and tap Rerun below."):
            if pending_box_full is None:
                st.warning("Draw or tap a box first.")
            else:
                st.session_state.box_list.append(pending_box_full)
                st.session_state.tap_corner1 = None
                st.session_state.tap_box_ready = None
                st.rerun()
        if clear_col.button("🗑️ Clear added boxes", use_container_width=True,
                            disabled=not st.session_state.box_list):
            st.session_state.box_list = []
            st.rerun()

        # The box she's currently drawn/tapped (if any) counts too, on top
        # of whatever's already in the list - so the LAST box doesn't need
        # its own separate "Add" tap before Rerun.
        all_boxes = list(st.session_state.box_list)
        if pending_box_full is not None:
            all_boxes = all_boxes + [pending_box_full]
        rerun_label = f"Rerun with {len(all_boxes)} boxes" if len(all_boxes) > 1 else "Rerun with this box"

        rerun_col, cancel_col = st.columns(2)
        if rerun_col.button(rerun_label, type="primary", use_container_width=True):
            if not all_boxes:
                st.warning("Draw or tap a box first.")
            else:
                with _shared_resource_lock:  # don't race a background prefetch using the same model
                    if len(all_boxes) == 1:
                        box = np.array(all_boxes[0], dtype=np.float32)
                        new_mask = rerun_with_box(medsam_model, st.session_state.image_tensor, box)
                        st.session_state.per_box_masks_result = None  # only one box - nothing extra to show
                    else:
                        # One shared image_encoder pass for every box, not one
                        # pass per box - see run_inference_multi_box's docstring.
                        new_mask, per_box_masks = rerun_with_boxes(
                            medsam_model, st.session_state.image_tensor, all_boxes)
                        st.session_state.per_box_masks_result = per_box_masks
                # Her corrected boxes, kept regardless of how many there were -
                # this is what later lets a YOLO fine-tuning pass learn from
                # her real corrections instead of only ever seeing its own
                # original (possibly wrong) guesses.
                st.session_state.last_boxes_used = all_boxes
                # Keep the old mask around so the comparison screen can show
                # what changed - without this, overwriting predicted_mask
                # straight away would mean there's nothing left to compare.
                st.session_state.mask_before_rerun = mask
                st.session_state.predicted_mask = new_mask
                put_in_session_cache(filename, (
                    st.session_state.image_tensor, st.session_state.image_resized_uint8,
                    original_image, new_mask,
                ))
                _, cached_mask_path = _disk_cache_paths(filename)
                if os.path.exists(os.path.dirname(cached_mask_path)):
                    np.save(cached_mask_path, new_mask)  # keep disk cache in sync with her correction
                st.session_state.tap_corner1 = None
                st.session_state.tap_box_ready = None
                st.session_state.box_list = []
                st.session_state.stage = "box_result"
                st.rerun()
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.tap_corner1 = None
            st.session_state.tap_box_ready = None
            st.session_state.box_list = []
            st.session_state.stage = "reviewing"
            st.rerun()

    # ---- Stage: box_result (compare old mask vs the box-rerun result) ----
    elif st.session_state.stage == "box_result":
        per_box_masks = st.session_state.get("per_box_masks_result")
        if per_box_masks:
            # Show what EACH box found on its own, before the union - so
            # it's visible that box 1 covered this tooth, box 2 covered that
            # one, etc., not just the final combined blob with no way to
            # tell which box contributed what.
            st.write(f"You marked {len(per_box_masks)} boxes - here's what each one found:")
            # A single row stretched across however many boxes there are
            # squeezes each image unreadably thin once she marks 5-6+ for a
            # full arch (Streamlit splits the row width evenly, no matter
            # how many columns). Wrapping into rows keeps each image a
            # sane size regardless of box count. Reuses the SAME "Smaller
            # canvas" checkbox she already toggles on the drawing screen,
            # rather than adding a second control for the same idea - she's
            # already told us whether she wants bigger/fewer-per-row images.
            per_row = 2 if st.session_state.get("box_small_canvas") else 3
            for row_start in range(0, len(per_box_masks), per_row):
                row_masks = per_box_masks[row_start:row_start + per_row]
                box_cols = st.columns(len(row_masks))
                for i, (col, box_mask) in enumerate(zip(box_cols, row_masks)):
                    with col:
                        st.image(overlay_mask(original_image, box_mask),
                                 caption=f"Box {row_start + i + 1}", use_container_width=True)
            st.write("Combined into one final mask:")

        st.write("Here's the result before and after your box(es).")
        before_col, after_col = st.columns(2)
        with before_col:
            st.image(overlay_mask(original_image, st.session_state.mask_before_rerun),
                     caption="Before (original prediction)", use_container_width=True)
        with after_col:
            st.image(overlay_mask(original_image, mask),
                     caption="After (your box(es), combined)", use_container_width=True)

        accept_col, redraw_col = st.columns(2)
        if accept_col.button("Use this result", type="primary", use_container_width=True):
            st.session_state.stage = "reviewing"
            st.rerun()
        if redraw_col.button("Draw the box again", use_container_width=True):
            st.session_state.stage = "box_prompt"
            st.rerun()

    # ---- Stage: editing (doctor redraws the mask by hand, zoomed in) ----
    elif st.session_state.stage == "editing":
        h, w = original_image.shape[:2]

        # Start each fresh editing session from the model's current
        # prediction (as plain tooth/background), not blank - she's
        # correcting what's there, not starting from scratch. Only reset
        # this when actually entering editing for a new image; otherwise
        # every rerun (every click) would wipe out work in progress.
        if st.session_state.get("editing_mask_filename") != filename:
            tooth_sized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            class_mask = np.where(tooth_sized > 0, MASK_CLASS_VALUES["tooth"], MASK_CLASS_VALUES["background"])
            st.session_state.editing_class_mask = class_mask.astype(np.uint8)
            st.session_state.editing_mask_filename = filename
            st.session_state.editing_zoom = 1.0
            st.session_state.editing_cx = 0.5
            st.session_state.editing_cy = 0.5
            st.session_state.editing_undo_stack = []  # snapshots taken before each Apply

        st.caption("Zoom in for fine detail - the brush stays the same size on "
                   "screen, but covers far fewer real pixels once you're zoomed in, "
                   "which is what gives finer control. Made a mistake while "
                   "drawing? The small toolbar in the top-right corner of the "
                   "canvas below has its own Undo - use that before tapping "
                   "'Apply this view'. Already applied a mistake? Use "
                   "'Undo last Apply' further down.")

        zoom_col, brush_col = st.columns(2)
        st.session_state.editing_zoom = zoom_col.slider(
            "Zoom", 1.0, 8.0, st.session_state.editing_zoom, step=0.25)
        brush_size = brush_col.slider("Brush size", 2, 60, BRUSH_SIZE)

        # Continuous sliders, not discrete preset buttons - a mouse/trackpad
        # drags a slider naturally to any position; the earlier 9-button
        # grid was built with touch in mind, which isn't how this is
        # actually being used (Dr. Sanskriti reviews on a PC, not a tablet).
        pos_col1, pos_col2 = st.columns(2)
        st.session_state.editing_cx = pos_col1.slider(
            "Left ↔ Right", 0.0, 1.0, st.session_state.editing_cx, step=0.02)
        st.session_state.editing_cy = pos_col2.slider(
            "Top ↔ Bottom", 0.0, 1.0, st.session_state.editing_cy, step=0.02)

        class_choice = st.radio(
            "What are you drawing:",
            ["🦷 Tooth", "⚠️ Ignore (overlap/ambiguous)", "🔴 Anomaly (flag for later)", "🧹 Erase"],
            horizontal=True,
        )
        class_key = {"🦷 Tooth": "tooth", "⚠️ Ignore (overlap/ambiguous)": "ignore",
                    "🔴 Anomaly (flag for later)": "anomaly", "🧹 Erase": "erase"}[class_choice]

        st.image(overlay_class_mask(original_image, st.session_state.editing_class_mask),
                 caption="Current progress (zoomed-out overview)", use_container_width=True)

        small_canvas = st.checkbox("Smaller canvas (easier for touch / Apple Pencil)",
                                   key="edit_small_canvas")
        max_dim = 420 if small_canvas else CANVAS_MAX_DIM

        zoom = st.session_state.editing_zoom
        cx, cy = st.session_state.editing_cx, st.session_state.editing_cy
        x0, y0, x1, y1 = view_window(h, w, zoom, cx, cy)
        crop = original_image[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape[:2]
        canvas_h, canvas_w, _ = scale_for_canvas(crop_h, crop_w, max_dim=max_dim)
        display_crop = cv2.resize(crop, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

        # Canvas key includes the window, not the class/brush choice - so
        # switching class or brush size keeps drawing on the SAME crop
        # (strokes accumulate), but changing zoom/position starts a fresh
        # blank canvas over the new crop, exactly as it visually should.
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=brush_size,
            stroke_color=f"rgb{CLASS_STROKE_COLORS[class_key]}",
            background_image=Image.fromarray(display_crop),
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="freedraw",
            # A newer streamlit-drawable-canvas release stopped returning
            # image_data by default (RuntimeError: "image_data was not
            # requested") - decode_class_canvas() below needs the raw pixel
            # data, not just json_data, so it has to be explicitly opted in.
            return_image_data=True,
            key=f"edit_canvas_{filename}_{zoom}_{cx}_{cy}",
        )

        apply_col, undo_col, done_col, cancel_col = st.columns(4)
        if apply_col.button("Apply this view", use_container_width=True,
                            help="Save what you've drawn here before zooming/moving elsewhere."):
            if canvas_result.image_data is not None:
                class_crop, touched_crop = decode_class_canvas(canvas_result.image_data)
                class_crop = cv2.resize(class_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                touched_crop = cv2.resize(touched_crop.astype(np.uint8), (crop_w, crop_h),
                                          interpolation=cv2.INTER_NEAREST) > 0
                if touched_crop.any():
                    # Snapshot BEFORE merging, not after - "Undo last Apply"
                    # needs to restore to how things looked before this
                    # Apply, so the snapshot has to be taken first. Capped at
                    # 10 so a long editing session can't grow this unbounded.
                    st.session_state.editing_undo_stack.append(st.session_state.editing_class_mask.copy())
                    st.session_state.editing_undo_stack = st.session_state.editing_undo_stack[-10:]
                    region = st.session_state.editing_class_mask[y0:y1, x0:x1]
                    region[touched_crop] = class_crop[touched_crop]
                    st.session_state.editing_class_mask[y0:y1, x0:x1] = region
                    st.success("Applied.")
                else:
                    st.warning("Draw something first.")
            else:
                st.warning("Draw something first.")
        if undo_col.button("↩️ Undo last Apply", use_container_width=True,
                           disabled=not st.session_state.editing_undo_stack):
            st.session_state.editing_class_mask = st.session_state.editing_undo_stack.pop()
            st.rerun()
        if done_col.button("Done - save as final", type="primary", use_container_width=True):
            # Her hand-drawn mark IS the ground truth once she taps this -
            # saved directly, no separate "are you sure?" screen in
            # between. She already reviewed it while drawing; a second
            # confirmation step on top of that was redundant friction, not
            # a real safeguard.
            mask_bytes = cv2.imencode(".png", st.session_state.editing_class_mask)[1].tobytes()
            mask_name = filename.rsplit(".", 1)[0] + "_mask.png"
            file_id = client.upload_bytes(mask_name, approved_folder_id, mask_bytes, "image/png")
            st.session_state.reviews_db = save_review(
                client, filename, "approved_after_edit", file_id, doctor_shade,
                corrected_boxes=st.session_state.get("last_boxes_used"))
            # Keep the in-session/disk-cached mask in sync with what was
            # just saved - without this, "What the model sees" would keep
            # showing the stale ORIGINAL prediction if she revisits this
            # photo later, even though her hand-edit is already the real
            # saved answer in Drive. overlay_mask() expects a plain
            # tooth/not-tooth mask, so collapse the 4-class edit down to
            # just the tooth class for that view - the class detail itself
            # still lives in the saved _mask.png, this is only the display
            # cache.
            tooth_only_mask = (st.session_state.editing_class_mask == MASK_CLASS_VALUES["tooth"]).astype(np.uint8)
            st.session_state.predicted_mask = tooth_only_mask
            put_in_session_cache(filename, (
                st.session_state.image_tensor, st.session_state.image_resized_uint8,
                original_image, tooth_only_mask,
            ))
            _, cached_mask_path = _disk_cache_paths(filename)
            if os.path.exists(os.path.dirname(cached_mask_path)):
                np.save(cached_mask_path, tooth_only_mask)
            _advance_after_review()
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.stage = "reviewing"
            st.rerun()

    # ---- Progress ----
    # Reuses the cached copy from earlier in this function - no need for a
    # fresh Drive round trip just to count what's already in memory and
    # correct as of the last save.
    db = st.session_state.reviews_db
    st.sidebar.metric("Reviewed", f"{len(db)} / {len(image_list)}")
    approved = sum(1 for r in db.values() if r["status"] in ("approved", "approved_after_edit"))
    needs_review = sum(1 for r in db.values() if r["status"] == "needs_review")
    st.sidebar.write(f"{STATUS_ICON['approved']} Approved: {approved}")
    st.sidebar.write(f"{STATUS_ICON['needs_review']} Needs review: {needs_review}")

    # ---- Stop instance (so she can end her own session) ----
    if LAMBDA_STARTER_URL:
        st.sidebar.divider()
        if st.sidebar.button("Done for now - stop instance", use_container_width=True):
            st.session_state.confirming_stop = True

        if st.session_state.get("confirming_stop"):
            st.sidebar.warning("This ends your session - you'll need to tap "
                               "Start again on the bookmark page next time.")
            confirm_col, cancel_col = st.sidebar.columns(2)
            if confirm_col.button("Yes, stop", use_container_width=True):
                try:
                    urllib.request.urlopen(f"{LAMBDA_STARTER_URL}?action=stop", timeout=10)
                    st.sidebar.success("Stopping - this page will stop responding shortly.")
                except Exception:
                    st.sidebar.error("Couldn't reach the stop switch - let the team know.")
                st.session_state.confirming_stop = False
            if cancel_col.button("Cancel", use_container_width=True):
                st.session_state.confirming_stop = False
                st.rerun()


def _go_to(index):
    st.session_state.current_index = index
    st.session_state.stage = "predicting"
    st.rerun()


def _advance_after_review():
    # Moves forward one image after a decision, same as flipping to the next
    # photo in the stack - stops at the end rather than wrapping, so she
    # gets a clear "that's everything" signal instead of looping forever.
    if st.session_state.current_index < len(st.session_state.image_list) - 1:
        st.session_state.current_index += 1
    st.session_state.stage = "predicting"
    st.rerun()


if __name__ == "__main__":
    main()
