"""Live Drowsiness Monitor - works on deployed apps (Snapshot Mode) and locally (WebRTC)."""
import os
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    GREEN,
    RED,
    best_model,
    find_eyes,
    gauge,
    get_theme,
    hero,
    inject_css,
    make_alarm_wav,
    render_footer,
    render_navbar,
    section_header,
    svg_icon,
    to_model_input,
)

st.set_page_config(page_title="DrowsAI | Live Monitor", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_navbar("monitor")
t = get_theme()

st.markdown(
    """
<style>
  .metric-card {
    padding: 22px 18px;
    border-radius: 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-soft);
    text-align: center;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .metric-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
  }
  .metric-value {
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
    font-size: 2.35rem;
    font-weight: 600;
    margin: 0;
    line-height: 1;
  }
  .metric-value.good { color: var(--good); }
  .metric-value.warning { color: var(--warning); }
  .metric-value.danger { color: var(--danger); }
  .metric-value.neutral { color: var(--neutral); }

  .big-eye-state {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 22px 24px;
    border-radius: 20px;
    font-family: var(--font-sans);
    font-size: clamp(1.5rem, 2.8vw, 2.15rem);
    font-weight: 600;
    letter-spacing: 0.02em;
    margin: 14px 0 18px 0;
  }
  .big-eye-state.open {
    background: var(--good-soft);
    color: var(--good);
    border: 1px solid var(--good);
  }
  .big-eye-state.closed {
    background: var(--danger-soft);
    color: var(--danger);
    border: 1px solid var(--danger);
  }
  .big-eye-state.noface {
    background: var(--surface-soft);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
  }
</style>
""",
    unsafe_allow_html=True,
)

hero(
    title="Continuous drowsiness monitor.",
    subtitle="Real-time eye-closure duration and PERCLOS tracking with optional acoustic alert. Supports snapshot capture everywhere and WebRTC streaming locally.",
    chips=["Snapshot & WebRTC Modes", "PERCLOS Windowing", "Acoustic Siren"],
    eyebrow="REAL-TIME TELEMETRY",
)

try:
    model, default_thr, cfg = best_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

WEBRTC_AVAILABLE = False
try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

mode = st.radio(
    "Select Monitoring Mode",
    ["Snapshot Mode (Deployed Apps & Mobile)", "Live Video Mode (Local WebRTC)"],
    horizontal=True,
    help="Snapshot mode works everywhere. Live video mode requires local streamlit-webrtc installation.",
)


def render_state_banner(state_label: str) -> None:
    if state_label == "CLOSED":
        cls, icon = "closed", svg_icon("eye-off", 26, 1.8)
    elif state_label == "OPEN":
        cls, icon = "open", svg_icon("eye", 26, 1.8)
    else:
        cls, icon = "noface", svg_icon("alert", 24, 1.8)
    st.markdown(f'<div class="big-eye-state {cls}">{icon}<span>{state_label}</span></div>', unsafe_allow_html=True)


def render_metrics_row(p: float, run: float, perclos: float, thr: float, closed_secs: float, perclos_limit: float, window_s: int) -> None:
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    p_class = "danger" if p >= thr else "good"
    m1.markdown(
        f'<div class="metric-card"><div class="metric-label">Closed Probability</div>'
        f'<p class="metric-value {p_class}">{p:.0%}</p></div>',
        unsafe_allow_html=True,
    )
    run_class = "danger" if run >= closed_secs else "good"
    m2.markdown(
        f'<div class="metric-card"><div class="metric-label">Closed Duration</div>'
        f'<p class="metric-value {run_class}">{run:.1f}s</p></div>',
        unsafe_allow_html=True,
    )
    perclos_class = "danger" if perclos >= perclos_limit else "good"
    m3.markdown(
        f'<div class="metric-card"><div class="metric-label">PERCLOS ({window_s}s)</div>'
        f'<p class="metric-value {perclos_class}">{perclos:.0%}</p></div>',
        unsafe_allow_html=True,
    )
    tint = t["danger"] if p >= thr else t["good"]
    with m4, st.container(border=True):
        st.markdown(gauge(p, 134, 12, "Score", tint), unsafe_allow_html=True)


# ------------------------------------------------------------------ Snapshot Mode
if mode.startswith("Snapshot"):
    with st.container(border=True):
        st.markdown("**Detection Settings**")
        c1, c2, c3, c4 = st.columns(4)
        thr = c1.slider("Closed-eye threshold", 0.05, 0.95, float(default_thr), 0.01)
        closed_secs = c2.slider("Alert if closed for (s)", 1.0, 5.0, 2.0, 0.5)
        window = c3.slider("PERCLOS window (s)", 5, 30, 10)
        perclos_limit = c4.slider("PERCLOS alert level", 0.2, 0.8, 0.4, 0.05)
        alarm_on = st.toggle("Enable acoustic alarm sound", value=True)

    section_header(
        eyebrow="CAPTURE",
        title="Take a driver snapshot",
        subtitle="Frame-by-frame evaluation computes closed probability, sustained closure duration, and rolling PERCLOS.",
    )

    camera_image = st.camera_input("Point camera at your face", label_visibility="collapsed")

    if camera_image is not None:
        img_array = np.array(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if "snapshot_history" not in st.session_state:
            st.session_state.snapshot_history = deque(maxlen=20)
        if "closed_since" not in st.session_state:
            st.session_state.closed_since = None

        boxes = find_eyes(rgb)
        if not boxes:
            st.warning("No face detected. Please adjust your camera angle and lighting.")
        else:
            crops = [to_model_input(rgb, b) for b in boxes]
            probs = model.predict(np.stack(crops), verbose=0).ravel()
            p = float(probs.mean())
            closed = p >= thr

            render_state_banner("CLOSED" if closed else "OPEN")

            now = time.time()
            st.session_state.snapshot_history.append((now, closed))

            recent = [(ts, c) for ts, c in st.session_state.snapshot_history if now - ts <= window]
            perclos = (sum(c for _, c in recent) / len(recent)) if recent else 0.0

            if closed:
                if st.session_state.closed_since is None:
                    st.session_state.closed_since = now
                run = now - st.session_state.closed_since
            else:
                st.session_state.closed_since = None
                run = 0.0

            render_metrics_row(p, run, perclos, thr, closed_secs, perclos_limit, int(window))

            alert = run >= closed_secs or (len(recent) >= 10 and perclos >= perclos_limit)
            if alert:
                st.markdown(
                    f'<div class="alert-calm">{svg_icon("alert", 22, 2.0)}'
                    f'<span>DROWSINESS DETECTED — Sustained eye closure exceeded threshold.</span></div>',
                    unsafe_allow_html=True,
                )
                if alarm_on:
                    st.audio(make_alarm_wav(), format="audio/wav", autoplay=True)

            if len(st.session_state.snapshot_history) > 1:
                st.write("")
                with st.container(border=True):
                    st.markdown("**Recent Snapshots**")
                    tokens = "".join(
                        f'<span class="hist-tok {"closed" if c else "open"}">'
                        f'<span class="hist-dot"></span>{"CLOSED" if c else "OPEN"}</span>'
                        for _, c in list(st.session_state.snapshot_history)[-12:]
                    )
                    st.markdown(f'<div class="hist-strip">{tokens}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Live Video Mode
else:
    if not WEBRTC_AVAILABLE:
        st.error("Live video mode requires the streamlit-webrtc package. Install it locally to stream video:")
        st.code("pip install streamlit-webrtc")
        st.info("Switch to Snapshot Mode above—it works seamlessly on deployed apps and mobile browsers.")
        render_footer()
        st.stop()

    DET_W = 480

    class Monitor(VideoProcessorBase):
        def __init__(self):
            self.model, self.overlay = None, False
            self.thr, self.closed_secs, self.window, self.perclos_limit = 0.5, 2.0, 10.0, 0.4
            self.lock = threading.Lock()
            self.latest, self.seen = None, None
            self.boxes, self.flags, self.preview = [], [], None
            self.hist, self.closed_since = deque(), None
            self.smoothed_p, self.state, self.flip = 0.0, "STARTING", 0
            self.last = dict(state="STARTING", p=0.0, perclos=0.0, run=0.0, alert=False)
            self.last_frame_t = time.time()
            threading.Thread(target=self._worker, daemon=True).start()

        def _worker(self):
            while time.time() - self.last_frame_t < 6:
                img = self.latest
                if img is None or img is self.seen or self.model is None:
                    time.sleep(0.01)
                    continue
                self.seen = img
                try:
                    self._analyse(img, time.time())
                except Exception:
                    time.sleep(0.05)

        def _analyse(self, img, now):
            h, w = img.shape[:2]
            scale = max(w / DET_W, 1.0)
            small = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA) if scale > 1 else img
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            sboxes = find_eyes(small_rgb)
            if not sboxes:
                self.boxes, self.flags, self.closed_since, self.smoothed_p, self.state = [], [], None, 0.0, "NO FACE"
                self.hist.clear()
                self.preview = small_rgb
                with self.lock:
                    self.last = dict(state="NO FACE", p=0.0, perclos=0.0, run=0.0, alert=False)
                return
            boxes = [(int(b[0] * scale), int(b[1] * scale), min(int(b[2] * scale), w), min(int(b[3] * scale), h)) for b in sboxes]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            crops = np.stack([to_model_input(rgb, b) for b in boxes])
            probs = np.asarray(self.model(crops, training=False)).ravel()
            self.smoothed_p = 0.5 * float(probs.mean()) + 0.5 * self.smoothed_p
            closed = self.smoothed_p >= self.thr
            if closed != (self.state == "CLOSED"):
                self.flip += 1
                if self.flip >= 2:
                    self.state, self.flip = ("CLOSED" if closed else "OPEN"), 0
            else:
                self.flip = 0
            self.boxes, self.flags = boxes, [float(x) >= self.thr for x in probs]
            prev = small_rgb.copy()
            for b, c in zip(sboxes, self.flags):
                cv2.rectangle(prev, (b[0], b[1]), (b[2], b[3]), RED if c else GREEN, 2)
            self.preview = prev

            self.hist.append((now, closed))
            while self.hist and now - self.hist[0][0] > self.window:
                self.hist.popleft()
            perclos = sum(c for _, c in self.hist) / len(self.hist)
            self.closed_since = (self.closed_since or now) if closed else None
            run = now - self.closed_since if self.closed_since else 0.0
            alert = run >= self.closed_secs or (len(self.hist) >= 15 and perclos >= self.perclos_limit)
            with self.lock:
                self.last = dict(state=self.state, p=self.smoothed_p, perclos=perclos, run=run, alert=alert)

        def recv(self, frame):
            self.last_frame_t = time.time()
            img = frame.to_ndarray(format="bgr24")
            self.latest = img
            if not self.overlay:
                return frame
            out = img.copy()
            for b, c in zip(self.boxes, self.flags):
                cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), RED[::-1] if c else GREEN[::-1], 3)
            with self.lock:
                s = dict(self.last)
            col = (0, 0, 255) if s["state"] == "CLOSED" else ((0, 200, 0) if s["state"] == "OPEN" else (190, 190, 190))
            cv2.putText(out, f"{s['state']}  P={s['p']:.2f}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(out, f"{s['state']}  P={s['p']:.2f}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            if s["alert"]:
                cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 255), 12)
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    with st.container(border=True):
        st.markdown("**Detection & Camera Settings**")
        c1, c2, c3, c4 = st.columns(4)
        thr = c1.slider("Closed-eye threshold", 0.05, 0.95, float(default_thr), 0.01)
        closed_secs = c2.slider("Alarm if closed for (s)", 1.0, 5.0, 2.0, 0.5)
        window = c3.slider("PERCLOS window (s)", 5, 30, 10)
        perclos_limit = c4.slider("PERCLOS alert level", 0.2, 0.8, 0.4, 0.05)

        cam_c1, cam_c2 = st.columns(2)
        camera_option = cam_c1.selectbox(
            "Select Camera",
            ["Auto (Default)", "Front Camera (Selfie)", "Back Camera (Environment)"],
            help="Choose which camera to use. Auto lets the browser select.",
        )
        mode_video = cam_c2.radio(
            "Video mode",
            ["Fast HD (recommended)", "Eye boxes drawn on video"],
            horizontal=True,
            help="Fast HD keeps your camera feed smooth in the browser without round-trip frame lag.",
        )
        fast = mode_video.startswith("Fast HD")
        alarm_on = st.toggle("Enable acoustic alarm sound", value=True, help="Plays siren alarm when drowsiness is detected")

    facing_mode = None
    if camera_option == "Front Camera (Selfie)":
        facing_mode = "user"
    elif camera_option == "Back Camera (Environment)":
        facing_mode = "environment"

    video_constraints = {
        "width": {"ideal": 1280 if fast else 640},
        "height": {"ideal": 720 if fast else 480},
        "frameRate": {"ideal": 30},
    }
    if facing_mode:
        video_constraints["facingMode"] = facing_mode

    ctx = webrtc_streamer(
        key="monitor_fast" if fast else "monitor_overlay",
        mode=WebRtcMode.SENDONLY if fast else WebRtcMode.SENDRECV,
        video_processor_factory=Monitor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": video_constraints, "audio": False},
    )

    status = st.empty()
    audio_counter = 0

    if ctx.video_processor:
        vp = ctx.video_processor
        vp.model, vp.overlay = model, (not fast)
        sound_on = False
        sound_ph = st.empty()

        while ctx.state.playing:
            vp.thr, vp.closed_secs, vp.window, vp.perclos_limit = thr, closed_secs, float(window), perclos_limit
            with vp.lock:
                s = dict(vp.last)
            prev = vp.preview

            with status.container():
                render_state_banner(s["state"])
                if fast and prev is not None:
                    with st.container(border=True):
                        st.image(prev, caption="AI Detection View", width="stretch")
                render_metrics_row(s["p"], s["run"], s["perclos"], thr, closed_secs, perclos_limit, int(window))

                if s["alert"]:
                    st.markdown(
                        f'<div class="alert-calm">{svg_icon("alert", 22, 2.0)}'
                        f'<span>DROWSINESS DETECTED — Sustained eye closure exceeded threshold.</span></div>',
                        unsafe_allow_html=True,
                    )
                elif s["state"] == "NO FACE":
                    st.warning("No face detected. Sit upright and face the camera with even lighting.")

            if s["alert"] and alarm_on and not sound_on:
                audio_counter += 1
                sound_ph.audio(make_alarm_wav(), format="audio/wav", loop=True, autoplay=True, key=f"alarm_{audio_counter}")
                sound_on = True
            elif sound_on and not (s["alert"] and alarm_on):
                sound_ph.empty()
                sound_on = False

            time.sleep(0.3)

        sound_ph.empty()
    else:
        st.info("Click START, grant camera permission, and position your face 40-60 cm from the lens with even lighting.")

render_footer()
