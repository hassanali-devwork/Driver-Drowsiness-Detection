"""Live Drowsiness Monitor - fast, HD, with a real alarm sound."""
import io, os, sys, threading, time, wave
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import inject_css, hero, best_model, find_eyes, to_model_input, GREEN, RED

st.set_page_config(page_title="Live Monitor", page_icon="🎥", layout="wide")
inject_css()

# The live video packages are optional: they are installed locally but NOT on the deployed app,
# so this page shows a friendly message instead of crashing when they are missing.
try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
except ImportError:
    st.title("🎥 Live Drowsiness Monitor")
    st.info("The live video monitor runs **locally** (it needs the `streamlit-webrtc` package and a direct camera connection).\n\n"
            "Run it on your own computer with:\n\n`pip install streamlit-webrtc`\n\n`streamlit run eye_app/app/Home.py`\n\n"
            "On this deployed app, use the **Home** page: take camera snapshots or upload a picture, "
            "and the snapshot history raises a drowsiness alert.")
    st.stop()
st.markdown("""
<style>
  .status-card {padding: 16px 18px; border-radius: 16px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
               border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 20px rgba(0,0,0,0.25); margin-bottom: 10px;}
  .status-card h3 {margin: 0 0 8px 0; font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: 1px; color: #e2e8f0;}
  .status-value {font-size: 1.9rem; font-weight: 800; margin: 0; line-height: 1;}
  .good {color: #22c55e;} .danger {color: #ef4444;} .neutral {color: #94a3b8;}
  .eye-indicator {display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px; border-radius: 999px;
                  font-weight: 800; font-size: 1.25rem; margin: 4px 0 10px 0;}
  .eye-indicator.open {background: rgba(34,197,94,.15); color: #22c55e; border: 2px solid #22c55e;}
  .eye-indicator.closed {background: rgba(239,68,68,.15); color: #ef4444; border: 2px solid #ef4444;}
  .eye-indicator.noface {background: rgba(148,163,184,.15); color: #94a3b8; border: 2px solid #94a3b8;}
  .alert-box {border-radius: 16px; padding: 22px; background: linear-gradient(135deg, #dc2626, #991b1b); border: 2px solid #ef4444;
              color: #fff; font-weight: 800; font-size: 1.4rem; text-align: center; margin-top: 8px;
              animation: pulse-alert 1s ease-in-out infinite; box-shadow: 0 8px 32px rgba(239,68,68,.5);}
  @media (max-width: 700px) {.status-value {font-size: 1.4rem;} .alert-box {font-size: 1.1rem; padding: 16px;}}
  @keyframes pulse-alert { 50% { transform: scale(1.02); box-shadow: 0 12px 40px rgba(239,68,68,.75); } }
</style>""", unsafe_allow_html=True)

hero("🎥 Live Drowsiness Monitor",
     "Continuous webcam monitoring. The eye-state CNN runs in the background; an alarm sounds when the eyes stay closed too long "
     "or the share of closed time (PERCLOS) gets too high.",
     ["⚡ HD video, no lag", "🔊 siren alarm", "📉 PERCLOS", "🧠 same CNN as the other pages"])

try:
    model, default_thr, cfg = best_model()
except Exception as e:
    st.error(f"Could not load the model: {e}"); st.stop()

DET_W = 480   # width of the copy used for face detection (fast); the CNN crops come from the full-resolution frame


@st.cache_data
def make_alarm_wav() -> bytes:
    """2.4 s wailing siren (650-1400 Hz sweep, rich harmonics), loops seamlessly."""
    sr, dur, period = 22050, 2.4, 0.8
    t = np.arange(int(sr * dur)) / sr
    tri = 1 - np.abs(2 * ((t % period) / period) - 1)
    phase = 2 * np.pi * np.cumsum(650 + 750 * tri) / sr
    phase -= (phase[-1] - 2 * np.pi * round(phase[-1] / (2 * np.pi))) * (t / dur)     # close the loop without a click
    sig = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    sig = sig / np.abs(sig).max() * 0.92
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((sig * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class Monitor(VideoProcessorBase):
    """recv() only stores the newest frame (and draws boxes in overlay mode). A worker thread does the AI work."""
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

    # ---------- background AI ----------
    def _worker(self):
        while time.time() - self.last_frame_t < 6:              # stops by itself when the stream ends
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
        crops = np.stack([to_model_input(rgb, b) for b in boxes])           # natural, un-filtered crops like the training data
        probs = np.asarray(self.model(crops, training=False)).ravel()
        self.smoothed_p = 0.5 * float(probs.mean()) + 0.5 * self.smoothed_p  # light smoothing against flicker
        closed = self.smoothed_p >= self.thr
        if closed != (self.state == "CLOSED"):                              # flip state only after 2 consistent readings
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

    # ---------- video thread (must stay light) ----------
    def recv(self, frame):
        self.last_frame_t = time.time()
        img = frame.to_ndarray(format="bgr24")
        self.latest = img
        if not self.overlay:
            return frame                                        # untouched frame, no extra work
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


# ------------------------------------------------------------------ settings
mode = st.radio("Video mode", ["⚡ Fast HD (recommended)", "🔲 Eye boxes drawn on the video"], horizontal=True,
                help="Fast HD keeps your camera picture in the browser (full quality, no lag) and shows the AI view next to it. "
                     "The other mode sends video through the server and back, which lowers quality and adds delay.")
fast = mode.startswith("⚡")
facing_label = st.radio("Camera", ["🤳 Front camera", "📷 Back camera"], horizontal=True,
                        help="On a phone you can choose the front (selfie) or back camera. On a laptop only one camera exists.")
facing = "user" if facing_label.startswith("🤳") else "environment"

c1, c2, c3, c4 = st.columns(4)
thr = c1.slider("Closed-eye threshold", 0.05, 0.95, float(default_thr), 0.01)
closed_secs = c2.slider("Alarm if closed for (s)", 1.0, 5.0, 2.0, 0.5)
window = c3.slider("PERCLOS window (s)", 5, 30, 10)
perclos_limit = c4.slider("PERCLOS alarm level", 0.2, 0.8, 0.4, 0.05)
s1, s2 = st.columns([1, 3])
alarm_on = s1.toggle("🔊 Alarm sound", value=True)
if s2.button("▶ Test alarm sound (also unlocks audio in your browser)"):
    st.audio(make_alarm_wav(), format="audio/wav", autoplay=True)

# ------------------------------------------------------------------ video
if fast:
    ctx = webrtc_streamer(key=f"monitor_fast_{facing}", mode=WebRtcMode.SENDONLY, video_processor_factory=Monitor, async_processing=True,
                          rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                          media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720},
                                                              "frameRate": {"ideal": 30}, "facingMode": {"ideal": facing}}, "audio": False})
else:
    ctx = webrtc_streamer(key=f"monitor_overlay_{facing}", mode=WebRtcMode.SENDRECV, video_processor_factory=Monitor, async_processing=True,
                          rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                          media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                                              "frameRate": {"ideal": 30}, "facingMode": {"ideal": facing}}, "audio": False})

# ------------------------------------------------------------------ live dashboard
status, sound_ph = st.empty(), st.empty()


def card(title, value, cls):
    return f'<div class="status-card"><h3>{title}</h3><p class="status-value {cls}">{value}</p></div>'


if ctx.video_processor:
    vp = ctx.video_processor
    vp.model, vp.overlay = model, (not fast)
    sound_on = False
    while ctx.state.playing:
        vp.thr, vp.closed_secs, vp.window, vp.perclos_limit = thr, closed_secs, float(window), perclos_limit
        with vp.lock:
            s = dict(vp.last)
        prev = vp.preview
        with status.container():
            left, right = st.columns([2, 3], gap="large")
            with left:
                if fast and prev is not None:
                    st.image(prev, caption="AI view (boxes = eyes found)", width="stretch")
            with right:
                cls = "closed" if s["state"] == "CLOSED" else ("open" if s["state"] == "OPEN" else "noface")
                emo = "😴" if s["state"] == "CLOSED" else ("👀" if s["state"] == "OPEN" else "❓")
                st.markdown(f'<div class="eye-indicator {cls}">{emo} {s["state"]}</div>', unsafe_allow_html=True)
                a, b, c = st.columns(3)
                a.markdown(card("Closed probability", f"{s['p']:.0%}", "danger" if s["p"] >= thr else "good"), unsafe_allow_html=True)
                b.markdown(card("Closed for", f"{s['run']:.1f}s", "danger" if s["run"] >= closed_secs else "good"), unsafe_allow_html=True)
                c.markdown(card(f"PERCLOS ({window}s)", f"{s['perclos']:.0%}", "danger" if s["perclos"] >= perclos_limit else "good"), unsafe_allow_html=True)
                if s["alert"]:
                    st.markdown('<div class="alert-box">🚨 DROWSINESS DETECTED - WAKE UP!</div>', unsafe_allow_html=True)
                elif s["state"] == "NO FACE":
                    st.warning("No face found - sit upright, face the camera, keep your whole face in view.")
        if s["alert"] and alarm_on and not sound_on:
            sound_ph.audio(make_alarm_wav(), format="audio/wav", loop=True, autoplay=True); sound_on = True
        elif sound_on and not (s["alert"] and alarm_on):
            sound_ph.empty(); sound_on = False
        time.sleep(0.3)
    sound_ph.empty()
else:
    st.info("Click **START**, allow camera access and face the camera. Good, even light on your face gives the best results.")