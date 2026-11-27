"""Eye-State Detector - main page. Run: streamlit run eye_app/app/Home.py"""
import hashlib
import time
import streamlit as st

from utils import (
    annotate,
    best_model,
    gauge,
    get_crops,
    get_theme,
    hero,
    inject_css,
    load_csv,
    predict,
    read_image,
    render_footer,
    render_navbar,
    section_header,
    svg_icon,
)

st.set_page_config(page_title="DrowsAI | Drowsiness Detector", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_navbar("home")

# ------------------------------------------------------------------ model & metrics
try:
    model, default_thr, cfg = best_model()
except Exception as e:
    st.error(f"Could not load models/best_model.keras: {e}")
    st.stop()

t = get_theme()
metrics = cfg.get("test_metrics", {})
comp = load_csv("phase2_model_comparison.csv")
latency_str = "1.29 ms"
if comp is not None and not comp.empty:
    row = comp[comp["model"] == cfg.get("model_name")]
    if not row.empty:
        latency_str = f"{float(row.iloc[0]['ms_per_image']):.2f} ms"

f1_val = float(metrics.get("f1_closed", 0.9745))
auc_val = float(metrics.get("auc", 0.9966))

# ------------------------------------------------------------------ hero
hero(
    title="Calm precision for every blink.",
    subtitle=(
        "Detects open and closed eye states in milliseconds from camera snapshots or uploaded "
        "frames—the core building block of a real-time driver drowsiness monitor."
    ),
    chips=[
        "Camera & Image Upload",
        f"Model: {cfg.get('model_name', 'ResNet-mini')}",
        "64x64 Grayscale CNN",
        "Subject-Wise Validated",
    ],
    eyebrow=f"MRL EYE DATASET  |  {cfg.get('model_name', 'RESNET-MINI').upper()}",
)

# ------------------------------------------------------------------ metric trio
m1, m2, m3 = st.columns(3, gap="medium")
with m1:
    st.markdown(
        f'<div class="calm-card">'
        f'<div class="stat-num">{f1_val:.1%}</div>'
        f'<div class="stat-title">Closed-Eye F1 Score</div>'
        f'<p class="stat-desc">Balanced precision and recall measured on held-out drivers never seen during training.</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f'<div class="calm-card">'
        f'<div class="stat-num">{auc_val:.1%}</div>'
        f'<div class="stat-title">Test ROC-AUC</div>'
        f'<p class="stat-desc">Consistent separation between open and closed eye crops across lighting and sensors.</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="calm-card">'
        f'<div class="stat-num">{latency_str}</div>'
        f'<div class="stat-title">Inference Per Eye</div>'
        f'<p class="stat-desc">Lightweight 64x64 grayscale evaluation engineered to run smoothly on standard CPUs.</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ interactive detector
section_header(
    eyebrow="LIVE EVALUATION",
    title="Inspect eye state in seconds",
    subtitle="Take a snapshot or upload a portrait. Use Settings below to adjust threshold sensitivity or switch to pre-cropped eye input.",
)

with st.expander("Detection Parameters", expanded=False):
    mode_label = st.radio(
        "Input type",
        ["Face photo (find eyes automatically)", "Eye close-up (use image as is)"],
        help="The model was trained on small eye crops. For a normal selfie or webcam photo use the first option.",
    )
    mode = "auto" if mode_label.startswith("Face") else "crop"
    sc1, sc2 = st.columns(2)
    thr = sc1.slider(
        "Closed-eye threshold",
        0.05,
        0.95,
        float(default_thr),
        0.01,
        help="Probability above which an eye is classified as CLOSED. Lower values increase sensitivity.",
    )
    alert_n = sc2.slider("Alert after N closed snapshots in a row", 2, 6, 3)
    st.caption(f"Active model: {cfg.get('model_name')}  |  Trained operating threshold: {default_thr:.2f}")


def show_result(rgb):
    ph = st.empty()
    ph.markdown('<div class="loader"><div class="spin"></div><p>ANALYSING FRAME</p></div>', unsafe_allow_html=True)
    crops, boxes = get_crops(rgb, mode)
    probs = predict(model, crops) if crops else []
    time.sleep(0.45)
    ph.empty()

    if not crops:
        st.warning("No eyes detected. Try a clearer, front-facing photo with even lighting—or switch to Eye Close-Up in Detection Parameters.")
        st.image(rgb, width="stretch")
        return None

    flags = [p >= thr for p in probs]
    p_mean = float(probs.mean())
    closed = p_mean >= thr
    tint = t["danger"] if closed else t["good"]

    left, right = st.columns([3, 2], gap="large")
    with left, st.container(border=True):
        st.image(
            annotate(rgb, boxes, flags) if mode == "auto" else rgb,
            width="stretch",
            caption="Green bounding box = open, red bounding box = closed" if mode == "auto" else "Input eye crop",
        )
    with right, st.container(border=True):
        st.markdown(gauge(p_mean, 210, 16, "Closed probability", tint, cls="big"), unsafe_allow_html=True)
        icon = svg_icon("eye-off" if closed else "eye", 18, 1.9)
        label = "EYES CLOSED" if closed else "EYES OPEN"
        st.markdown(
            f'<div class="verdict-pill {"closed" if closed else "open"}">{icon}<span>{label}</span></div>'
            f'<div style="text-align:center;color:var(--text-muted);font-size:0.84rem;margin-bottom:8px;">'
            f'Decision threshold: {thr:.0%}</div>',
            unsafe_allow_html=True,
        )
        if len(probs) > 1:
            eyes_html = "".join(
                f'<div style="text-align:center;">'
                f'{gauge(p, 92, 8, "", t["danger"] if p >= thr else t["good"])}'
                f'<div style="font-size:0.78rem;color:var(--text-secondary);margin-top:-2px;">'
                f'Eye {i + 1}: {"Closed" if p >= thr else "Open"}</div></div>'
                for i, p in enumerate(probs)
            )
            st.markdown(f'<div style="display:flex;justify-content:center;gap:18px;margin-top:8px;">{eyes_html}</div>', unsafe_allow_html=True)
        with st.expander("Model Input Preview (64x64 Grayscale)"):
            st.image([c[..., 0] for c in crops], width=108, clamp=True)
    return closed


tab_cam, tab_up = st.tabs(["Selfie Camera", "Phone Camera / Upload"])

with tab_cam:
    st.caption("Capture consecutive snapshots to test the multi-frame drowsiness alert. On mobile, use the Upload tab to access the rear camera.")
    if "history" not in st.session_state:
        st.session_state.history, st.session_state.last = [], None
    shot = st.camera_input("Look at the camera", label_visibility="collapsed")
    if shot is not None:
        res = show_result(read_image(shot))
        h = hashlib.md5(shot.getvalue()).hexdigest()
        if res is not None and h != st.session_state.last:
            st.session_state.history.append(bool(res))
            st.session_state.last = h

    hist = st.session_state.history
    if hist:
        with st.container(border=True):
            st.markdown("**Snapshot Sequence** (oldest to latest)")
            tokens = "".join(
                f'<span class="hist-tok {"closed" if c else "open"}">'
                f'<span class="hist-dot"></span>'
                f'#{idx + 1} {"CLOSED" if c else "OPEN"}</span>'
                for idx, c in enumerate(hist[-12:])
            )
            st.markdown(f'<div class="hist-strip">{tokens}</div>', unsafe_allow_html=True)
            if len(hist) >= alert_n and all(hist[-alert_n:]):
                st.markdown(
                    f'<div class="alert-calm">{svg_icon("alert", 20, 1.9)}'
                    f'<span>DROWSINESS ALERT — Eyes closed in the last {alert_n} consecutive snapshots.</span></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Reset Snapshot History"):
                st.session_state.history, st.session_state.last = [], None
                st.rerun()

with tab_up:
    st.info(
        "On mobile devices, tap Browse files and choose Take Photo to switch between front and rear cameras. "
        "On desktop, upload any JPG or PNG portrait or eye close-up."
    )
    up = st.file_uploader("Upload or capture a photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if up is not None:
        show_result(read_image(up))
    else:
        st.caption("Default mode detects eyes automatically inside a face photo. Switch to Eye Close-Up inside Detection Parameters for cropped eye images.")

# ------------------------------------------------------------------ architecture flow cards
section_header(
    eyebrow="PIPELINE ARCHITECTURE",
    title="Three calm stages from frame to alert",
    subtitle="Every prediction follows the same deterministic preprocessing pipeline validated during Phase 2 modeling.",
)

p1, p2, p3 = st.columns(3, gap="medium")
with p1:
    st.markdown(
        f"""
        <div class="calm-card">
          <div class="preview-box" style="align-items:center;">
            <div style="display:flex;gap:18px;align-items:center;">
              <div style="width:48px;height:48px;border-radius:10px;border:1.5px solid var(--good);display:flex;align-items:center;justify-content:center;color:var(--good);font-family:var(--font-mono);font-size:0.7rem;">L-EYE</div>
              <div style="width:48px;height:48px;border-radius:10px;border:1.5px solid var(--good);display:flex;align-items:center;justify-content:center;color:var(--good);font-family:var(--font-mono);font-size:0.7rem;">R-EYE</div>
            </div>
            <div style="margin-top:10px;font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);">Haar Frontal Face Geometry</div>
          </div>
          <div class="card-icon-wrap">{svg_icon("camera", 17)}</div>
          <div class="card-h">Reliable Eye Localization</div>
          <p class="card-p">Detects the frontal face first, then derives proportional square eye boxes so closed eyelids are never skipped.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p2:
    st.markdown(
        f"""
        <div class="calm-card">
          <div class="preview-box" style="align-items:center;">
            <div style="font-family:var(--font-mono);font-size:0.86rem;color:var(--text);font-weight:500;">64 x 64 x 1</div>
            <div style="width:82%;height:6px;border-radius:999px;background:var(--border);margin:10px 0;overflow:hidden;">
              <div style="width:76%;height:100%;background:var(--accent);border-radius:999px;"></div>
            </div>
            <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);">Replicate Pad + [0, 1] Scale</div>
          </div>
          <div class="card-icon-wrap">{svg_icon("layers", 17)}</div>
          <div class="card-h">Normalized Grayscale Input</div>
          <p class="card-p">Aspect-preserving border padding and bilinear resizing align live camera frames with the infrared training distribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p3:
    st.markdown(
        f"""
        <div class="calm-card">
          <div class="preview-box" style="align-items:center;">
            <div style="display:flex;gap:6px;align-items:center;">
              <span style="width:10px;height:10px;border-radius:50%;background:var(--good);"></span>
              <span style="width:10px;height:10px;border-radius:50%;background:var(--good);"></span>
              <span style="width:10px;height:10px;border-radius:50%;background:var(--danger);"></span>
              <span style="width:10px;height:10px;border-radius:50%;background:var(--danger);"></span>
              <span style="width:10px;height:10px;border-radius:50%;background:var(--danger);"></span>
            </div>
            <div style="margin-top:12px;font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);">Consecutive Closure + PERCLOS</div>
          </div>
          <div class="card-icon-wrap">{svg_icon("activity", 17)}</div>
          <div class="card-h">Temporal Fatigue Logic</div>
          <p class="card-p">Separates natural blinks from sustained drowsiness using consecutive closed frames and rolling PERCLOS thresholds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ closing CTA banner
st.markdown(
    """
    <div class="cta-banner">
      <div class="hero-eyebrow" style="margin-bottom:10px;">CONTINUE EXPLORING</div>
      <h3>Inspect model benchmarks or launch live monitoring</h3>
      <p>Review failure modes across glasses and lighting on the Model Results page, or open the Live Monitor for continuous PERCLOS tracking and acoustic alerts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
c_left, c_mid1, c_mid2, c_right = st.columns([1.2, 1.5, 1.5, 1.2])
with c_mid1:
    st.page_link("pages/1_Model_Results.py", label="Open Model Results", use_container_width=True)
with c_mid2:
    st.page_link("pages/2_Live_Monitor.py", label="Open Live Monitor", use_container_width=True)

render_footer()
