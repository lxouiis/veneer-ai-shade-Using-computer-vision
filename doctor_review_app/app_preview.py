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
                           help="Draw directly on the mask with a brush."):
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
                st.success("(Preview) Would rerun the model with this box - mask unchanged here.")
                st.session_state.stage = "reviewing"
                st.rerun()
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.stage = "reviewing"
            st.rerun()

    elif st.session_state.stage == "editing":
        st.info("Draw over the tooth region to correct the mask. White = tooth, "
                "use the eraser tool to remove incorrect areas.")
        h, w = original_image.shape[:2]
        canvas_h, canvas_w, _ = scale_for_canvas(h, w)
        display_image = cv2.resize(original_image, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=BRUSH_SIZE,
            stroke_color="white",
            background_image=Image.fromarray(display_image),
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="freedraw",
            key=f"edit_canvas_{filename}",
        )

        save_col, cancel_col = st.columns(2)
        if save_col.button("Save corrected mask", type="primary", use_container_width=True):
            if canvas_result.image_data is not None:
                st.session_state.stage = "confirming"
                st.rerun()
            else:
                st.warning("Draw the corrected tooth region before saving.")
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.stage = "reviewing"
            st.rerun()

    elif st.session_state.stage == "confirming":
        st.image(overlay_mask(original_image, mask), caption="Corrected mask (preview - same fake mask shown)",
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
