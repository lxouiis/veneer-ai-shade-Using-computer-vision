"""Frontend-only preview of app.py - same UI layout and click-through flow
(navigation bar, box-redraw tool, hand-correction tool), but with a
synthetic photo and synthetic mask instead of real Drive/MedSAM/YOLO calls.
Zero setup: no checkpoints, no Google Drive, no AWS.

Run with: streamlit run app_preview.py
"""
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

MASK_OVERLAY_COLOR = (255, 0, 0)
MASK_OVERLAY_ALPHA = 0.4
BRUSH_SIZE = 12
CANVAS_MAX_DIM = 900
STATUS_ICON = {"not_reviewed": "⬜", "approved": "✅", "needs_review": "\U0001f6a9"}
VITA_SHADES = ["A1", "A2", "A3", "A3.5", "A4", "B1", "B2", "B3", "B4",
               "C1", "C2", "C3", "C4", "D2", "D3", "D4"]
NOT_SURE = "-- not sure / skip --"

MASK_CLASS_VALUES = {"background": 0, "anomaly": 85, "ignore": 170, "tooth": 255}
CLASS_STROKE_COLORS = {
    "tooth": (255, 255, 255), "ignore": (255, 165, 0),
    "anomaly": (255, 0, 255), "erase": (0, 0, 0),
}
CLASS_OVERLAY_COLORS = {"tooth": (0, 255, 0), "ignore": (255, 165, 0), "anomaly": (255, 0, 255)}
CENTER_PRESETS = {
    "↖": (0.2, 0.2), "↑": (0.5, 0.2), "↗": (0.8, 0.2),
    "←": (0.2, 0.5), "•": (0.5, 0.5), "→": (0.8, 0.5),
    "↙": (0.2, 0.8), "↓": (0.5, 0.8), "↘": (0.8, 0.8),
}

FAKE_FILENAMES = ["fake_patient_001.jpg", "fake_patient_002.jpg", "fake_patient_003.jpg"]


def make_fake_image(seed):
    h, w = 600, 800
    image = np.full((h, w, 3), (235, 215, 205), dtype=np.uint8)
    cx = w // 2 + (seed * 37 % 60 - 30)  # slight variation per "image" so Prev/Next looks different
    cv2.ellipse(image, (cx, h // 2), (180, 90), 0, 0, 360, (250, 245, 235), -1)
    return image


def make_fake_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (150, 60), 0, 0, 360, 1, -1)
    return mask


def overlay_mask(image, mask):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)
    color_layer[mask_resized > 0] = MASK_OVERLAY_COLOR
    blended = overlay * (1 - MASK_OVERLAY_ALPHA) + color_layer * MASK_OVERLAY_ALPHA
    return blended.astype(np.uint8)


def overlay_class_mask(image, class_mask):
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
    h, w = image_data.shape[:2]
    class_values = np.zeros((h, w), dtype=np.uint8)
    touched = np.zeros((h, w), dtype=bool)
    alpha = image_data[:, :, 3]
    rgb = image_data[:, :, :3]
    for class_name, stroke_color in CLASS_STROKE_COLORS.items():
        close = np.all(np.abs(rgb.astype(np.int16) - np.array(stroke_color)) < 30, axis=-1)
        hit = close & (alpha > 0)
        class_values[hit] = MASK_CLASS_VALUES.get(class_name, MASK_CLASS_VALUES["background"])
        touched |= hit
    return class_values, touched


def view_window(h, w, zoom, cx, cy):
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


def load_image(index):
    st.session_state.original_image = make_fake_image(index)
    st.session_state.predicted_mask = make_fake_mask(st.session_state.original_image.shape)


def main():
    st.set_page_config(page_title="Tooth Mask Review (PREVIEW)", layout="wide")
    st.title("Tooth Segmentation Review")
    st.caption("PREVIEW MODE - fake photos, nothing real is loaded or saved.")

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
        st.session_state.stage = "reviewing"
        st.session_state.statuses = {f: "not_reviewed" for f in FAKE_FILENAMES}
        load_image(0)

    index = st.session_state.current_index
    filename = FAKE_FILENAMES[index]
    original_image = st.session_state.original_image
    mask = st.session_state.predicted_mask
    current_status = st.session_state.statuses[filename]

    nav_prev, nav_pos, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("← Previous", use_container_width=True, disabled=index == 0):
            _go_to(index - 1)
    with nav_pos:
        st.markdown(
            f"<div style='text-align:center'>{STATUS_ICON[current_status]} "
            f"<b>Image {index + 1} / {len(FAKE_FILENAMES)}</b> &mdash; {filename}</div>",
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("Next →", use_container_width=True, disabled=index == len(FAKE_FILENAMES) - 1):
            _go_to(index + 1)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original photo", use_container_width=True)
    with col2:
        st.image(overlay_mask(original_image, mask), caption="What the model sees", use_container_width=True)

    shade_choice = st.selectbox(
        "What VITA shade do you think this actually is? (your own judgement, not the model's guess)",
        [NOT_SURE] + VITA_SHADES, key=f"shade_{filename}",
    )
    doctor_shade = None if shade_choice == NOT_SURE else shade_choice  # noqa: F841 (preview only, not saved anywhere)

    if st.session_state.stage == "reviewing":
        approve_col, box_col, hand_col = st.columns(3)
        if approve_col.button("✅ Looks good", type="primary", use_container_width=True):
            st.session_state.statuses[filename] = "approved"
            st.success("(Preview) Would upload the mask and mark this approved.")

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

    elif st.session_state.stage == "box_prompt":
        st.info("Draw a box around the tooth - click and drag one rectangle, then tap Rerun.")
        canvas_h, canvas_w, scale = scale_for_canvas(*original_image.shape[:2])
        display_frame = cv2.resize(original_image, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

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
                # Preview has no real model - fake a visibly different mask
                # (shifted oval) purely so the before/after screen has
                # something distinct to show side by side.
                st.session_state.mask_before_rerun = mask
                shifted = make_fake_mask(original_image.shape)
                shifted = np.roll(shifted, 40, axis=1)
                st.session_state.predicted_mask = shifted
                st.session_state.stage = "box_result"
                st.rerun()
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.stage = "reviewing"
            st.rerun()

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

    elif st.session_state.stage == "editing":
        h, w = original_image.shape[:2]

        if st.session_state.get("editing_mask_filename") != filename:
            tooth_sized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            class_mask = np.where(tooth_sized > 0, MASK_CLASS_VALUES["tooth"], MASK_CLASS_VALUES["background"])
            st.session_state.editing_class_mask = class_mask.astype(np.uint8)
            st.session_state.editing_mask_filename = filename
            st.session_state.editing_zoom = 1.0
            st.session_state.editing_cx = 0.5
            st.session_state.editing_cy = 0.5
            st.session_state.editing_undo_stack = []

        st.caption("Zoom in for fine detail. The canvas below has its own built-in "
                   "Undo in its top-right toolbar; already applied a mistake? Use "
                   "'Undo last Apply' further down.")

        zoom_col, brush_col = st.columns(2)
        st.session_state.editing_zoom = zoom_col.slider(
            "Zoom", 1.0, 6.0, st.session_state.editing_zoom, step=0.5)
        brush_size = brush_col.slider("Brush size", 2, 60, BRUSH_SIZE)

        st.write("Which part of the photo:")
        preset_cols = st.columns(3)
        for i, (label, (cx, cy)) in enumerate(CENTER_PRESETS.items()):
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
        if apply_col.button("Apply this view", use_container_width=True):
            if canvas_result.image_data is not None:
                class_crop, touched_crop = decode_class_canvas(canvas_result.image_data)
                class_crop = cv2.resize(class_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                touched_crop = cv2.resize(touched_crop.astype(np.uint8), (crop_w, crop_h),
                                          interpolation=cv2.INTER_NEAREST) > 0
                if touched_crop.any():
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

    elif st.session_state.stage == "confirming":
        st.image(overlay_class_mask(original_image, st.session_state.corrected_mask),
                 caption="Corrected mask (green=tooth, orange=ignore, magenta=anomaly)",
                 use_container_width=True)
        st.write("Is this corrected mask now accurate?")
        yes_col, no_col = st.columns(2)
        if yes_col.button("Yes - looks correct now", type="primary", use_container_width=True):
            st.session_state.statuses[filename] = "approved"
            st.success("(Preview) Would upload the corrected mask and mark this approved.")
            st.session_state.stage = "reviewing"
        if no_col.button("No - still wrong, flag for review", use_container_width=True):
            st.session_state.statuses[filename] = "needs_review"
            st.warning("(Preview) Would flag this image for review.")
            st.session_state.stage = "reviewing"

    approved = sum(1 for s in st.session_state.statuses.values() if s == "approved")
    needs_review = sum(1 for s in st.session_state.statuses.values() if s == "needs_review")
    reviewed = approved + needs_review
    st.sidebar.metric("Reviewed", f"{reviewed} / {len(FAKE_FILENAMES)}")
    st.sidebar.write(f"{STATUS_ICON['approved']} Approved: {approved}")
    st.sidebar.write(f"{STATUS_ICON['needs_review']} Needs review: {needs_review}")
    st.sidebar.divider()
    st.sidebar.button("Done for now - stop instance", use_container_width=True, disabled=True,
                       help="Preview only - this does nothing here. Real version calls the Lambda starter.")


def _go_to(index):
    st.session_state.current_index = index
    st.session_state.stage = "reviewing"
    load_image(index)
    st.rerun()


if __name__ == "__main__":
    main()
