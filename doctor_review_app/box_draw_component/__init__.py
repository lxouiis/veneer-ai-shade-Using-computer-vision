"""A small, dependency-free box-drawing Streamlit component, built from
scratch after streamlit-drawable-canvas's drag-to-box tool was confirmed
broken on touch (box draws visually, never reaches Python - a live,
still-open upstream bug: github.com/andfanilo/streamlit-drawable-canvas
issue #144). No Node/React/build step - the frontend is plain HTML/JS,
same minimal pattern streamlit-image-coordinates itself uses, built on the
Pointer Events API (pointerdown/move/up) which unifies mouse, touch, and
pen/stylus input into one event model - the actual fix for the problem,
not another library with the same mouse-only event split underneath.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image

_frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component("box_draw_component", path=str(_frontend_dir))


def box_draw_component(image: np.ndarray, key: str | None = None):
    """Shows `image` (an RGB numpy array) and lets the user drag a box over
    it with mouse, touch, or stylus alike. Returns a dict with x1/y1/x2/y2
    (in the image's own pixel coordinates) once a box has been drawn, or
    None if nothing's been drawn yet."""
    pil_image = Image.fromarray(image)
    buf = BytesIO()
    pil_image.save(buf, format="PNG", compress_level=1)
    src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    h, w = image.shape[:2]
    return _component_func(src=src, width=w, height=h, key=key, default=None)
