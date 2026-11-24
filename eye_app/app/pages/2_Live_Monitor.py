"""Live Drowsiness Monitor - Works on local and deployed/mobile apps."""
import io
import os
import sys
import threading
import time
import wave
import warnings
from collections import deque
from pathlib import Path

# Suppress warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import inject_css, hero, best_model, find_eyes, to_model_input, GREEN, RED

st.set_page_config(page_title="Live Monitor", page_icon="🎥", layout="wide")
inject_css()

# CSS for both modes
st.markdown("""
<style>
  .block-container {padding-top: 1rem; max-width: 1400px;}

  .hero-section {padding: 32px 36px; border-radius: 24px; margin-bottom: 24px; color: #fff;
                background: linear-gradient(135deg, #6366f1 0%, #4338ca 50%, #0f172a 100%);
                box-shadow: 0 12px 40px rgba(67,56,202,.35);}
  .hero-section h1 {margin: 0 0 8px 0; font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; color: #fff;}
  .hero-section p {margin: 0; font-size: 1.15rem; opacity: 0.9; line-height: 1.5;}

  .main-status {padding: 28px 32px; border-radius: 20px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 2px solid rgba(255,255,255,0.15); box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 20px;}
  .main-status h2 {margin: 0 0 16px 0; font-size: 1rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 1.5px; color: #e2e8f0;}

  .big-eye-state {display: flex; align-items: center; justify-content: center; gap: 16px; padding: 24px;
                  border-radius: 18px; font-size: 2.5rem; font-weight: 800; text-align: center; margin: 16px 0;}
  .big-eye-state.open {background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1));
                       color: #22c55e; border: 3px solid #22c55e; box-shadow: 0 8px 24px rgba(34,197,94,0.3);}
  .big-eye-state.closed {background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.1));
                         color: #ef4444; border: 3px solid #ef4444; box-shadow: 0 8px 24px rgba(239,68,68,0.3);}
  .big-eye-state.noface {background: linear-gradient(135deg, rgba(148,163,184,0.2), rgba(148,163,184,0.1));
                         color: #94a3b8; border: 3px solid #94a3b8;}

  .metric-card {padding: 24px; border-radius: 18px; background: rgba(255,255,255,0.05);
               border: 1px solid rgba(255,255,255,0.1); text-align: center;}
  .metric-label {font-size: 0.85rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; color: #cbd5e1;}
  .metric-value {font-size: 2.8rem; font-weight: 800; margin: 0; line-height: 1;}
  .metric-value.good {color: #22c55e;}
  .metric-value.warning {color: #f59e0b;}
  .metric-value.danger {color: #ef4444;}
  .metric-value.neutral {color: #94a3b8;}

  .alert-banner {border-radius: 20px; padding: 32px; background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
                 border: 3px solid #ef4444; color: #fff; font-weight: 800; font-size: 2rem; text-align: center;
                 margin-top: 24px; animation: pulse-alert 1.2s ease-in-out infinite; box-shadow: 0 12px 48px rgba(239,68,68,0.6);}
  @keyframes pulse-alert { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.03); } }

  .settings-section {padding: 20px 24px; border-radius: 16px; background: rgba(255,255,255,0.03);
                      border: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;}
  .settings-section h3 {margin: 0 0 16px 0; font-size: 1.1rem; color: #e2e8f0;}

  .gauge-container {display: flex; flex-direction: column; align-items: center; padding: 20px; border-radius: 18px;
                    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);}
  @property --p { syntax: '<number>'; inherits: true; initial-value: 0; }
  @property --n { syntax: '<integer>'; inherits: true; initial-value: 0; }
  .gauge {position: relative; width: var(--size, 160px); height: var(--size, 160px); margin: 0 auto;
          animation: fill 1s cubic-bezier(.2,.8,.2,1) forwards;}
  @keyframes fill { to { --p: var(--target); --n: var(--target-int); } }
  .gauge .ring {position: absolute; inset: 0; border-radius: 50%;
          background: conic-gradient(var(--tint) calc(var(--p) * 1%), rgba(128,128,128,.2) 0);
          -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - var(--thick, 14px)), #000 calc(100% - var(--thick, 14px) + 1px));
                  mask: radial-gradient(farthest-side, transparent calc(100% - var(--thick, 14px)), #000 calc(100% - var(--thick, 14px) + 1px));
          filter: drop-shadow(0 0 10px var(--tint));}
  .gauge .gnum {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;}
  .gauge .pct::after {counter-reset: pct var(--n); content: counter(pct) "%"; font-weight: 800;
          font-size: calc(var(--size, 160px) * .22); line-height: 1; color: var(--tint);}
  .gauge small {margin-top: 6px; font-size: calc(var(--size, 160px) * .055); letter-spacing: 1px; opacity: 0.6; text-transform: uppercase;}

  @media (max-width: 768px) {
    .hero-section h1 {font-size: 2rem;}
    .big-eye-state {font-size: 1.8rem; padding: 16px;}
    .metric-value {font-size: 2rem;}
    .alert-banner {font-size: 1.4rem; padding: 20px;}
  }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-section">
  <h1>🎥 Live Drowsiness Monitor</h1>
  <p>Real-time AI-powered eye detection with automatic alarms. Works on desktop and mobile devices.</p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model, default_thr, cfg = best_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# Check if WebRTC is available (for local use)
WEBRTC_AVAILABLE = False
try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Mode selection
mode = st.radio(
    "Select Monitoring Mode",
    ["📸 Snapshot Mode (Works on deployed apps & mobile)", "🎥 Live Video Mode (Local only)"],
    help="Snapshot mode works everywhere. Live video mode requires local installation."
)

# Snapshot Mode (works on deployed apps and mobile)
if mode.startswith("📸"):
    st.markdown('<div class="settings-section"><h3>⚙️ Detection Settings</h3></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    thr = c1.slider("Closed-eye threshold", 0.05, 0.95, float(default_thr), 0.01)
    closed_secs = c2.slider("Alert if closed for (s)", 1.0, 5.0, 2.0, 0.5)
    window = c3.slider("PERCLOS window (s)", 5, 30, 10)
    perclos_limit = c4.slider("PERCLOS alert level", 0.2, 0.8, 0.4, 0.05)

    alarm_on = st.toggle("🔊 Enable alarm sound", value=True)

    st.divider()
    st.subheader("📷 Take Snapshot")

    # Camera input with mobile support
    camera_image = st.camera_input("Point camera at your face", label_visibility="collapsed")

    if camera_image is not None:
        # Process image
        import hashlib
        img_array = np.array(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Initialize session state
        if "snapshot_history" not in st.session_state:
            st.session_state.snapshot_history = deque(maxlen=20)
        if "closed_since" not in st.session_state:
            st.session_state.closed_since = None

        # Find eyes
        boxes = find_eyes(rgb)
        if not boxes:
            st.warning("⚠️ No face detected. Please adjust your position.")
        else:
            # Get crops and predict
            crops = [to_model_input(rgb, b) for b in boxes]
            probs = model.predict(np.stack(crops), verbose=0).ravel()
            p = float(probs.mean())
            closed = p >= thr

            # Display result
            state_class = "closed" if closed else "open"
            emoji = "😴" if closed else "👀"
            st.markdown(f'<div class="big-eye-state {state_class}">{emoji} {"CLOSED" if closed else "OPEN"}</div>', unsafe_allow_html=True)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            p_class = "danger" if p >= thr else "good"
            m1.markdown(f'<div class="metric-card"><div class="metric-label">Closed Probability</div><p class="metric-value {p_class}">{p:.0%}</p></div>', unsafe_allow_html=True)

            # Track history
            now = time.time()
            st.session_state.snapshot_history.append((now, closed))

            # Calculate PERCLOS
            recent = [(t, c) for t, c in st.session_state.snapshot_history if now - t <= window]
            if recent:
                perclos = sum(c for _, c in recent) / len(recent)
                perclos_class = "danger" if perclos >= perclos_limit else "good"
                m3.markdown(f'<div class="metric-card"><div class="metric-label">PERCLOS</div><p class="metric-value {perclos_class}">{perclos:.0%}</p></div>', unsafe_allow_html=True)

            # Calculate closed duration
            if closed:
                if st.session_state.closed_since is None:
                    st.session_state.closed_since = now
                run = now - st.session_state.closed_since
            else:
                st.session_state.closed_since = None
                run = 0.0

            run_class = "danger" if run >= closed_secs else "good"
            m2.markdown(f'<div class="metric-card"><div class="metric-label">Closed Duration</div><p class="metric-value {run_class}">{run:.1f}s</p></div>', unsafe_allow_html=True)

            # Gauge
            tint = "#ef4444" if p >= thr else "#22c55e"
            pct = max(0.0, min(100.0, p * 100))
            m4.markdown(f'<div class="gauge-container"><div class="gauge" style="--size:160px;--thick:14px;--tint:{tint};--target:{pct:.1f};--target-int:{round(pct)};"><div class="ring"></div><div class="gnum"><span class="pct"></span><small>Score</small></div></div></div>', unsafe_allow_html=True)

            # Alert
            alert = run >= closed_secs or (len(recent) >= 10 and perclos >= perclos_limit)
            if alert:
                st.markdown('<div class="alert-banner">🚨 DROWSINESS DETECTED - WAKE UP!</div>', unsafe_allow_html=True)
                if alarm_on:
                    st.audio(make_alarm_wav(), format="audio/wav", autoplay=True)

            # Show history
            if len(st.session_state.snapshot_history) > 1:
                st.subheader("📊 Recent Snapshots")
                history_emojis = " ".join(["😴" if c else "👀" for _, c in list(st.session_state.snapshot_history)[-12:]])
                st.markdown(f"<div style='font-size: 2rem; text-align: center; padding: 20px;'>{history_emojis}</div>", unsafe_allow_html=True)

# Live Video Mode (local only with WebRTC)
else:
    if not WEBRTC_AVAILABLE:
        st.error("❌ Live video mode requires `streamlit-webrtc` package. Please install it:")
        st.code("pip install streamlit-webrtc")
        st.info("Use **Snapshot Mode** above - it works on deployed apps and mobile devices!")
        st.stop()

    DET_W = 480

    @st.cache_data
    def make_alarm_wav() -> bytes:
        """2.4 s wailing siren alarm."""
        sr, dur, period = 22050, 2.4, 0.8
        t = np.arange(int(sr * dur)) / sr
        tri = 1 - np.abs(2 * ((t % period) / period) - 1)
        phase = 2 * np.pi * np.cumsum(650 + 750 * tri) / sr
        phase -= (phase[-1] - 2 * np.pi * round(phase[-1] / (2 * np.pi))) * (t / dur)
        sig = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
        sig = sig / np.abs(sig).max() * 0.92
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((sig * 32767).astype(np.int16).tobytes())
        return buf.getvalue()

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
                    time.sleep(0.01); continue
                self.seen = img
                try:
                    self._analyse(img, time.time())
                except Exception:
                    time.sleep(0.05)

        def _analyse(self, img, now):
            h, w = img.shape[:2]; scale = max(w / DET_W, 1.0)
            small = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA) if scale > 1 else img
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            sboxes = find_eyes(small_rgb)
            if not sboxes:
                self.boxes, self.flags, self.closed_since, self.smoothed_p, self.state = [], [], None, 0.0, "NO FACE"
                self.hist.clear(); self.preview = small_rgb
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
                if self.flip >= 2: self.state, self.flip = ("CLOSED" if closed else "OPEN"), 0
            else:
                self.flip = 0
            self.boxes, self.flags = boxes, [float(x) >= self.thr for x in probs]
            prev = small_rgb.copy()
            for b, c in zip(sboxes, self.flags):
                cv2.rectangle(prev, (b[0], b[1]), (b[2], b[3]), RED if c else GREEN, 2)
            self.preview = prev

            self.hist.append((now, closed))
            while self.hist and now - self.hist[0][0] > self.window: self.hist.popleft()
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

    st.markdown('<div class="settings-section"><h3>⚙️ Detection Settings</h3></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    thr = c1.slider("Closed-eye threshold", 0.05, 0.95, float(default_thr), 0.01)
    closed_secs = c2.slider("Alarm if closed for (s)", 1.0, 5.0, 2.0, 0.5)
    window = c3.slider("PERCLOS window (s)", 5, 30, 10)
    perclos_limit = c4.slider("PERCLOS alert level", 0.2, 0.8, 0.4, 0.05)

    st.markdown('<div class="settings-section"><h3>📷 Camera Settings</h3></div>', unsafe_allow_html=True)

    camera_option = st.selectbox(
        "Select Camera",
        ["Auto (Default)", "Front Camera (Selfie)", "Back Camera (Environment)"],
        help="Choose which camera to use. 'Auto' lets the browser decide."
    )

    facing_mode = None
    if camera_option == "Front Camera (Selfie)":
        facing_mode = "user"
    elif camera_option == "Back Camera (Environment)":
        facing_mode = "environment"

    mode_video = st.radio("Video mode", [" Fast HD (recommended)", "🔲 Eye boxes drawn on video"], horizontal=True,
                    help="Fast HD keeps your camera picture in the browser (full quality, no lag).")
    fast = mode_video.startswith(" Fast HD")

    alarm_on = st.toggle("🔊 Enable alarm sound", value=True, help="Plays siren alarm when drowsiness is detected")

    video_constraints = {
        "width": {"ideal": 1280 if fast else 640},
        "height": {"ideal": 720 if fast else 480},
        "frameRate": {"ideal": 30}
    }

    if facing_mode:
        video_constraints["facingMode"] = facing_mode

    if fast:
        ctx = webrtc_streamer(
            key="monitor_fast",
            mode=WebRtcMode.SENDONLY,
            video_processor_factory=Monitor,
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": video_constraints, "audio": False}
        )
    else:
        ctx = webrtc_streamer(
            key="monitor_overlay",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=Monitor,
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": video_constraints, "audio": False}
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
                st.markdown('<div class="main-status"><h2>📊 Live Detection Status</h2></div>', unsafe_allow_html=True)

                state_class = "closed" if s["state"] == "CLOSED" else ("open" if s["state"] == "OPEN" else "noface")
                emoji = "😴" if s["state"] == "CLOSED" else ("👀" if s["state"] == "OPEN" else "❓")
                st.markdown(f'<div class="big-eye-state {state_class}">{emoji} {s["state"]}</div>', unsafe_allow_html=True)

                if fast and prev is not None:
                    st.image(prev, caption="AI Detection View", width="stretch")

                st.markdown('<div class="main-status"><h2>📈 Detection Metrics</h2></div>', unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)

                p_class = "danger" if s['p'] >= thr else "good"
                m1.markdown(f'<div class="metric-card"><div class="metric-label">Closed Probability</div><p class="metric-value {p_class}">{s["p"]:.0%}</p></div>', unsafe_allow_html=True)

                run_class = "danger" if s['run'] >= closed_secs else "good"
                m2.markdown(f'<div class="metric-card"><div class="metric-label">Closed Duration</div><p class="metric-value {run_class}">{s["run"]:.1f}s</p></div>', unsafe_allow_html=True)

                perclos_class = "danger" if s['perclos'] >= perclos_limit else "good"
                m3.markdown(f'<div class="metric-card"><div class="metric-label">PERCLOS ({window}s)</div><p class="metric-value {perclos_class}">{s["perclos"]:.0%}</p></div>', unsafe_allow_html=True)

                tint = "#ef4444" if s['p'] >= thr else "#22c55e"
                pct = max(0.0, min(100.0, s['p'] * 100))
                m4.markdown(f'<div class="gauge-container"><div class="gauge" style="--size:160px;--thick:14px;--tint:{tint};--target:{pct:.1f};--target-int:{round(pct)};"><div class="ring"></div><div class="gnum"><span class="pct"></span><small>Score</small></div></div></div>', unsafe_allow_html=True)

                if s["alert"]:
                    st.markdown('<div class="alert-banner">🚨 DROWSINESS DETECTED - WAKE UP!</div>', unsafe_allow_html=True)
                elif s["state"] == "NO FACE":
                    st.warning("⚠️ No face detected. Please sit upright and face the camera with good lighting.")

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
        st.info("Click **START**, allow camera access and face the camera. Good, even light on your face gives the best results.")
        st.caption("💡 Tip: Position your face 40-60 cm from the camera with even lighting for best detection accuracy.")
        st.caption("📱 Mobile users: Use the camera selector above to choose front or back camera.")