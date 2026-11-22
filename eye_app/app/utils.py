"""Shared helpers for the Eye-State Streamlit app."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR, RESULTS_DIR, IMG_DIR = ROOT / "models", ROOT / "results", ROOT / "images"
IMG_SIZE = 64          # must match Phase 1 / Phase 2 preprocessing
GREEN, RED = (46, 160, 67), (220, 53, 69)


def inject_css():
    """Shared modern styling used by every page."""
    st.markdown("""
    <style>
      @property --p { syntax: '<number>';  inherits: true; initial-value: 0; }
      @property --n { syntax: '<integer>'; inherits: true; initial-value: 0; }
      .block-container {padding-top: 1.2rem; max-width: 1150px;}
      [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 20px;}

      .hero {padding: 26px 30px; border-radius: 24px; margin-bottom: 18px; color: #fff;
             background: radial-gradient(circle at 12% 20%, #6366f1 0%, #4338ca 38%, #0f172a 100%);
             box-shadow: 0 14px 40px rgba(67,56,202,.35);}
      .hero h1 {margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.5px; color: #fff;}
      .hero p {margin: 6px 0 0 0; font-size: 1.02rem; opacity: .88;}
      .chips {margin-top: 14px;}
      .chip {display: inline-block; padding: 4px 12px; margin: 0 6px 4px 0; border-radius: 999px; font-size: .8rem;
             background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);}

      .gauge {position: relative; width: var(--size, 210px); height: var(--size, 210px); margin: 6px auto;
              animation: fill 1.6s cubic-bezier(.2,.8,.2,1) forwards;}
      @keyframes fill { to { --p: var(--target); --n: var(--target-int); } }
      .gauge .ring {position: absolute; inset: 0; border-radius: 50%;
              background: conic-gradient(var(--tint) calc(var(--p) * 1%), rgba(128,128,128,.22) 0);
              -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - var(--thick, 16px)), #000 calc(100% - var(--thick, 16px) + 1px));
                      mask: radial-gradient(farthest-side, transparent calc(100% - var(--thick, 16px)), #000 calc(100% - var(--thick, 16px) + 1px));
              filter: drop-shadow(0 0 10px var(--tint));}
      .gauge .gnum {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;}
      .gauge .pct::after {counter-reset: pct var(--n); content: counter(pct) "%"; font-weight: 800;
              font-size: calc(var(--size, 210px) * .24); line-height: 1; color: var(--tint);}
      .gauge small {margin-top: 6px; font-size: calc(var(--size, 210px) * .065); letter-spacing: 1.5px; opacity: .65; text-transform: uppercase;}

      .alert {border-radius: 14px; padding: 14px 18px; background: #fdecea; border: 2px solid #e74c3c; color: #922b21;
              font-weight: 700; margin-top: 10px;}
    </style>""", unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", chips=()):
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><div class="chips">{chips_html}</div></div>',
                unsafe_allow_html=True)


def gauge(p, size=210, thick=16, label="", tint=None):
    """Animated circular gauge (HTML string) that counts from 0% up to p (0-1)."""
    pct = max(0.0, min(100.0, p * 100))
    tint = tint or ("#ef4444" if p >= 0.5 else "#22c55e")
    return (f'<div class="gauge" style="--size:{size}px;--thick:{thick}px;--tint:{tint};'
            f'--target:{pct:.1f};--target-int:{round(pct)};">'
            f'<div class="ring"></div><div class="gnum"><span class="pct"></span><small>{label}</small></div></div>')


# ---------- files, config, models ----------
@st.cache_resource
def load_config():
    p = MODEL_DIR / "phase2_config.json"
    return json.load(open(p)) if p.exists() else {"model_name": "best_model", "threshold": 0.5}


@st.cache_resource(show_spinner="Loading model ...")
def load_keras(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def best_model():
    cfg = load_config()
    return load_keras(str(MODEL_DIR / "best_model.keras")), float(cfg["threshold"]), cfg


def load_csv(name: str):
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def available_models():
    """{model name: (path, threshold)} for every deep model that has a saved .keras file."""
    comp, out = load_csv("phase2_model_comparison.csv"), {}
    if comp is not None:
        for _, r in comp[comp.family == "Deep learning"].iterrows():
            f = MODEL_DIR / f"p2_{r.model.replace(' ', '_').replace('(', '').replace(')', '')}.keras"
            if f.exists():
                out[r.model] = (str(f), float(r.threshold))
    if not out and (MODEL_DIR / "best_model.keras").exists():
        cfg = load_config()
        out[cfg["model_name"]] = (str(MODEL_DIR / "best_model.keras"), float(cfg["threshold"]))
    return out


# ---------- image handling ----------
def read_image(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    img.thumbnail((1280, 1280))
    return np.array(img)


@st.cache_resource
def _cascades():
    base = cv2.data.haarcascades
    return (cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(base + "haarcascade_eye.xml"))


def _square(x, y, w, h, W, H, pad=0.25):
    s = max(w, h) * (1 + pad); cx, cy = x + w / 2, y + h / 2
    return (int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0)), int(min(cx + s / 2, W)), int(min(cy + s / 2, H)))


def find_eyes(rgb):
    """Find the face with a Haar cascade, then place two square eye boxes from face geometry.
    (The Haar *eye* cascade is trained on open eyes and often misses closed ones, so it is not used.)"""
    face_c, _ = _cascades()
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)); H, W = gray.shape
    faces = face_c.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return []
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    side = 0.36 * fw
    boxes = []
    for cx in (fx + 0.30 * fw, fx + 0.70 * fw):                        # left / right eye centre
        cy = fy + 0.39 * fh
        boxes.append((int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)), int(min(cx + side / 2, W)), int(min(cy + side / 2, H))))
    return boxes


def to_model_input(rgb, box=None):
    """Grayscale -> (square) crop -> 64x64 -> [0,1]. Same recipe as the training data."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if box is not None:
        x0, y0, x1, y1 = box; gray = gray[y0:y1, x0:x1]
    h, w = gray.shape
    if h != w:                                                          # pad to square, keep the aspect ratio
        d = abs(h - w); a, b = d // 2, d - d // 2
        gray = cv2.copyMakeBorder(gray, a, b, 0, 0, cv2.BORDER_REPLICATE) if w > h else cv2.copyMakeBorder(gray, 0, 0, a, b, cv2.BORDER_REPLICATE)
    gray = np.asarray(Image.fromarray(gray).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)   # same as Phase 1
    return (gray.astype("float32") / 255.0)[..., None]


def get_crops(rgb, mode):
    """mode: 'auto' (find eyes in a face photo) or 'crop' (image already is an eye)."""
    if mode == "crop":
        return [to_model_input(rgb)], [None]
    boxes = find_eyes(rgb)
    return [to_model_input(rgb, b) for b in boxes], boxes


def predict(model, crops):
    return model.predict(np.stack(crops), verbose=0).ravel() if crops else np.array([])


def annotate(rgb, boxes, closed_flags):
    out = rgb.copy()
    for b, c in zip(boxes, closed_flags):
        if b is not None:
            cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), RED if c else GREEN, max(2, rgb.shape[1] // 200))
    return out