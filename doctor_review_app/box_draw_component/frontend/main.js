// Box-drawing component built on the Pointer Events API (pointerdown /
// pointermove / pointerup), a mature W3C standard specifically designed to
// unify mouse, touch, AND pen/stylus input into ONE consistent event
// model - unlike mouse-only event chains (mousedown/mousemove/mouseup),
// which is what left streamlit-drawable-canvas's drawn shapes never
// reaching Python on touch devices (confirmed: the shape rendered
// visually via Fabric.js's own touch-agnostic rendering, but the
// sync-back-to-Streamlit step never fired for a touch-originated drag).
// This component has no such split - the SAME pointerup handler that
// finalizes the box ALSO sends it to Python, for every input type alike.

const bg = document.getElementById("bg");
const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");

let drawing = false;
let startX = 0, startY = 0;
let lastBox = null;

function drawRect(x0, y0, x1, y1) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#ffd700";
  ctx.lineWidth = 3;
  ctx.fillStyle = "rgba(255, 215, 0, 0.15)";
  const x = Math.min(x0, x1), y = Math.min(y0, y1);
  const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0);
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);
}

function sendBox(x0, y0, x1, y1) {
  const box = {
    x1: Math.round(Math.min(x0, x1)), y1: Math.round(Math.min(y0, y1)),
    x2: Math.round(Math.max(x0, x1)), y2: Math.round(Math.max(y0, y1)),
    width: canvas.width, height: canvas.height,
    unix_time: Date.now(),
  };
  lastBox = box;
  Streamlit.setComponentValue(box);
}

function pointInCanvas(event) {
  const rect = canvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

canvas.addEventListener("pointerdown", (event) => {
  // Pointer capture - keeps delivering move/up events to THIS element even
  // if the finger/pen drifts outside its bounds mid-drag, which touch
  // input does far more often than a precise mouse. Without this, a drag
  // that wanders off the canvas edge would silently stop tracking.
  canvas.setPointerCapture(event.pointerId);
  const [x, y] = pointInCanvas(event);
  startX = x;
  startY = y;
  drawing = true;
  event.preventDefault();
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  const [x, y] = pointInCanvas(event);
  drawRect(startX, startY, x, y);
  event.preventDefault();
});

canvas.addEventListener("pointerup", (event) => {
  if (!drawing) return;
  drawing = false;
  const [x, y] = pointInCanvas(event);
  drawRect(startX, startY, x, y);
  sendBox(startX, startY, x, y);
  event.preventDefault();
});

canvas.addEventListener("pointercancel", () => {
  drawing = false;
});

function onRender(event) {
  const { src, width, height } = event.detail.args;

  if (bg.src !== src) {
    bg.src = src;
  }

  bg.onload = () => {
    const w = width || bg.naturalWidth;
    const h = height || bg.naturalHeight;
    bg.width = w;
    bg.height = h;
    canvas.width = w;
    canvas.height = h;
    document.getElementById("wrap").style.width = w + "px";
    document.getElementById("wrap").style.height = h + "px";
    Streamlit.setFrameHeight(h);
    // Redraw the last-sent box, if any, so it survives a Streamlit rerun
    // triggered by something unrelated (e.g. a slider elsewhere on the
    // page) instead of visually vanishing.
    if (lastBox) {
      drawRect(lastBox.x1, lastBox.y1, lastBox.x2, lastBox.y2);
    }
  };
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
