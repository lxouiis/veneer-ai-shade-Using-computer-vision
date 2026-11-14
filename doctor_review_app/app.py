import os
import sys
import json
import time
import tempfile
import urllib.request
import numpy as np
import torch
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

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
# other and from typical tooth/gum photo colors, so decoding a drawn stroke
# back to a class by its RGB value (see decode_class_canvas) is unambiguous.
CLASS_STROKE_COLORS = {
    "tooth": (255, 255, 255),      # white
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
# Zoom presets - a simple 3x3 "which part of the photo" picker rather than
# raw pan sliders. (cx, cy) are the fraction-of-image center point used by
# view_window below; CENTER_PRESETS' 9 options cover corners/edges/middle,
# enough to reach any part of a tooth photo without fiddly slider dragging.
CENTER_PRESETS = {
    "↖": (0.2, 0.2), "↑": (0.5, 0.2), "↗": (0.8, 0.2),
    "←": (0.2, 0.5), "•": (0.5, 0.5), "→": (0.8, 0.5),
    "↙": (0.2, 0.8), "↓": (0.5, 0.8), "↘": (0.8, 0.8),
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


def save_review(client, filename, status, mask_file_id=None, doctor_shade=None):
    db, reviews_file_id = load_reviews_db(client)
    db[filename] = {
        "status": status,  # "approved" | "approved_after_edit" | "needs_review"
        "mask_file_id": mask_file_id,
        # Dr. Sanskriti's own real shade judgement for this photo - separate
        # from, and stronger than, module_6_shade_classification's ΔE guess.
        # None means she didn't pick one (left on "not sure / skip") -
        # distinct from an actual answer, not treated as a guess either way.
        "doctor_shade": doctor_shade,
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


def scale_for_canvas(h, w):
    scale = min(1.0, CANVAS_MAX_DIM / max(h, w))
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
    st.title("Tooth Segmentation Review")
    st.caption("YOLO box + fine-tuned MedSAM (frozen encoder) - the same "
               "pipeline module_3 uses on Kaggle.")

    medsam_model, yolo_model = load_models()
    client, images_folder_id, approved_folder_id, needs_review_folder_id = get_drive()
    image_list = get_image_list(client, images_folder_id)
    reviews_db, _ = load_reviews_db(client)

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
        image_bytes = client.download_bytes(image_entry["id"])
        image_tensor, image_resized_uint8, original_image, mask = predict_mask(
            image_bytes, filename, medsam_model, yolo_model
        )
        st.session_state.image_tensor = image_tensor
        st.session_state.image_resized_uint8 = image_resized_uint8
        st.session_state.original_image = original_image
        st.session_state.predicted_mask = mask
        st.session_state.last_prediction_filename = filename
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
            save_review(client, filename, "approved", file_id, doctor_shade)
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
        st.info("Draw a box around the tooth - click and drag one rectangle, "
                "then tap Rerun.")

        canvas_h, canvas_w, scale = scale_for_canvas(*st.session_state.image_resized_uint8.shape[:2])
        display_frame = cv2.resize(st.session_state.image_resized_uint8, (canvas_w, canvas_h),
                                   interpolation=cv2.INTER_AREA)

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

        rerun_col, cancel_col = st.columns(2)
        if rerun_col.button("Rerun with this box", type="primary", use_container_width=True):
            rect = rect_from_canvas(canvas_result)
            if rect is None:
                st.warning("Draw a box first.")
            else:
                # Canvas was shown scaled down from the 1024x1024 working
                # frame - scale the box back up to that same space, which is
                # exactly what the model expects as a prompt.
                box = np.array([c / scale for c in rect], dtype=np.float32)
                new_mask = rerun_with_box(medsam_model, st.session_state.image_tensor, box)
                # Keep the old mask around so the comparison screen can show
                # what changed - without this, overwriting predicted_mask
                # straight away would mean there's nothing left to compare.
                st.session_state.mask_before_rerun = mask
                st.session_state.predicted_mask = new_mask
                st.session_state.stage = "box_result"
                st.rerun()
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.stage = "reviewing"
            st.rerun()

    # ---- Stage: box_result (compare old mask vs the box-rerun result) ----
    elif st.session_state.stage == "box_result":
        st.write("Here's the result before and after your box.")
        before_col, after_col = st.columns(2)
        with before_col:
            st.image(overlay_mask(original_image, st.session_state.mask_before_rerun),
                     caption="Before (original prediction)", use_container_width=True)
        with after_col:
            st.image(overlay_mask(original_image, mask),
                     caption="After (your box)", use_container_width=True)

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
            "Zoom", 1.0, 6.0, st.session_state.editing_zoom, step=0.5)
        brush_size = brush_col.slider("Brush size", 2, 60, BRUSH_SIZE)

        st.write("Which part of the photo:")
        preset_cols = st.columns(3)
        presets = list(CENTER_PRESETS.items())
        for i, (label, (cx, cy)) in enumerate(presets):
            if preset_cols[i % 3].button(label, use_container_width=True, key=f"center_{label}_{filename}"):
                st.session_state.editing_cx, st.session_state.editing_cy = cx, cy
                st.rerun()

        class_choice = st.radio(
            "What are you drawing:",
            ["🦷 Tooth", "⚠️ Ignore (overlap/ambiguous)", "🔴 Anomaly (flag for later)", "🧹 Erase"],
            horizontal=True,
        )
        class_key = {"🦷 Tooth": "tooth", "⚠️ Ignore (overlap/ambiguous)": "ignore",
                    "🔴 Anomaly (flag for later)": "anomaly", "🧹 Erase": "erase"}[class_choice]

        st.image(overlay_class_mask(original_image, st.session_state.editing_class_mask),
                 caption="Current progress (zoomed-out overview)", use_container_width=True)

        zoom = st.session_state.editing_zoom
        cx, cy = st.session_state.editing_cx, st.session_state.editing_cy
        x0, y0, x1, y1 = view_window(h, w, zoom, cx, cy)
        crop = original_image[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape[:2]
        canvas_h, canvas_w, _ = scale_for_canvas(crop_h, crop_w)
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
        if done_col.button("Done - review result", type="primary", use_container_width=True):
            st.session_state.corrected_mask = st.session_state.editing_class_mask
            st.session_state.stage = "confirming"
            st.rerun()
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.stage = "reviewing"
            st.rerun()

    # ---- Stage: confirming (is the corrected mask now right?) ----
    elif st.session_state.stage == "confirming":
        st.image(
            overlay_class_mask(original_image, st.session_state.corrected_mask),
            caption="Corrected mask (green=tooth, orange=ignore, magenta=anomaly)",
            use_container_width=True,
        )
        st.write("Is this corrected mask now accurate?")
        yes_col, no_col = st.columns(2)

        if yes_col.button("Yes - looks correct now", type="primary", use_container_width=True):
            mask_bytes = cv2.imencode(".png", st.session_state.corrected_mask)[1].tobytes()
            mask_name = filename.rsplit(".", 1)[0] + "_mask.png"
            file_id = client.upload_bytes(mask_name, approved_folder_id, mask_bytes, "image/png")
            save_review(client, filename, "approved_after_edit", file_id, doctor_shade)
            _advance_after_review()

        if no_col.button("No - still wrong, flag for review", use_container_width=True):
            flagged_bytes = cv2.imencode(".jpg", cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))[1].tobytes()
            client.upload_bytes(filename, needs_review_folder_id, flagged_bytes, "image/jpeg")
            save_review(client, filename, "needs_review", doctor_shade=doctor_shade)
            _advance_after_review()

    # ---- Progress ----
    db, _ = load_reviews_db(client)
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
