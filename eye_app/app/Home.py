"""Eye-State Detector - main page.  Run:  streamlit run app/Home.py"""
import hashlib
import time
import streamlit as st
from utils import inject_css, best_model, read_image, get_crops, predict, annotate

st.set_page_config(page_title="Eye-State Detector", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")
inject_css()

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
  @property --p { syntax: '<number>';  inherits: true; initial-value: 0; }
  @property --n { syntax: '<integer>'; inherits: true; initial-value: 0; }

  .block-container {padding-top: 1.2rem; max-width: 1150px;}
  [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 20px;}

  /* ---------- hero ---------- */
  .hero {padding: 26px 30px; border-radius: 24px; margin-bottom: 18px; color: #fff;
         background: radial-gradient(circle at 12% 20%, #6366f1 0%, #4338ca 38%, #0f172a 100%);
         box-shadow: 0 14px 40px rgba(67,56,202,.35);}
  .hero h1 {margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.5px; color: #fff;}
  .hero p {margin: 6px 0 0 0; font-size: 1.02rem; opacity: .88;}
  .chips {margin-top: 14px;}
  .chip {display: inline-block; padding: 4px 12px; margin: 0 6px 4px 0; border-radius: 999px; font-size: .8rem;
         background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);}

  /* ---------- animated gauge ---------- */
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

  /* ---------- loading wheel ---------- */
  .loader {display: flex; flex-direction: column; align-items: center; padding: 26px 0;}
  .spin {width: 120px; height: 120px; border-radius: 50%; border: 12px solid rgba(128,128,128,.22);
         border-top-color: #6366f1; border-right-color: #8b5cf6; animation: rot .8s linear infinite;}
  @keyframes rot { to { transform: rotate(360deg); } }
  .loader p {margin-top: 14px; opacity: .7; letter-spacing: 1px;}

  /* ---------- verdict + misc ---------- */
  .pill {display: block; text-align: center; padding: 12px 18px; border-radius: 14px; font-size: 1.5rem; font-weight: 800;
         color: #fff; margin: 8px 0 4px 0; animation: pop .5s ease both;}
  .pill.closed {background: linear-gradient(135deg, #dc2626, #f43f5e); box-shadow: 0 8px 24px rgba(220,38,38,.4);}
  .pill.open {background: linear-gradient(135deg, #059669, #34d399); box-shadow: 0 8px 24px rgba(5,150,105,.4);}
  @keyframes pop { from {transform: scale(.85); opacity: 0;} to {transform: scale(1); opacity: 1;} }
  .sub {text-align: center; opacity: .7; font-size: .9rem; margin-bottom: 6px;}
  .eyes {display: flex; justify-content: center; gap: 18px; margin-top: 6px;}
  .eyes .gauge {margin: 0;}
  .eyecap {text-align: center; font-size: .8rem; opacity: .7; margin-top: -2px;}
  .hist span {display: inline-block; font-size: 1.5rem; margin-right: 4px;}
  .alert {border-radius: 14px; padding: 14px 18px; background: #fdecea; border: 2px solid #e74c3c; color: #922b21;
          font-weight: 700; margin-top: 10px; animation: pulse 1s ease-in-out infinite;}
  @keyframes pulse { 50% { box-shadow: 0 0 0 8px rgba(231,76,60,.18); } }
  /* ---------- phones / small screens ---------- */
  @media (max-width: 700px) {
    .block-container {padding: .7rem .75rem 3rem !important;}
    .hero {padding: 16px 16px; border-radius: 18px; margin-bottom: 12px;}
    .hero h1 {font-size: 1.5rem;} .hero p {font-size: .88rem;}
    .chip {font-size: .68rem; padding: 3px 9px;}
    .gauge.big {--size: 170px !important; --thick: 14px !important;}
    .pill {font-size: 1.15rem; padding: 10px 12px;}
    .eyes {gap: 10px;}
    .stButton > button, [data-testid="stFileUploader"] button {min-height: 3rem; font-size: 1rem;}
    button[data-baseweb="tab"] {padding-left: 10px; padding-right: 10px; font-size: .9rem;}
  }
</style>
""", unsafe_allow_html=True)


def gauge(p, size=210, thick=16, label="Closed probability", tint=None, cls=""):
    """Animated circular gauge, counts from 0% up to p."""
    pct = max(0.0, min(100.0, p * 100))
    tint = tint or ("#ef4444" if p >= 0.5 else "#22c55e")
    return (f'<div class="gauge {cls}" style="--size:{size}px;--thick:{thick}px;--tint:{tint};'
            f'--target:{pct:.1f};--target-int:{round(pct)};">'
            f'<div class="ring"></div><div class="gnum"><span class="pct"></span><small>{label}</small></div></div>')


# ------------------------------------------------------------------ model
try:
    model, default_thr, cfg = best_model()
except Exception as e:
    st.error(f"Could not load models/best_model.keras: {e}")
    st.stop()

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero">
  <h1>👁️ Eye-State Detector</h1>
  <p>Detects whether eyes are <b>open or closed</b> - the building block of a driver drowsiness monitor.</p>
  <div class="chips"><span class="chip">📷 Camera</span><span class="chip">🖼️ Upload</span>
  <span class="chip">🧠 {cfg.get('model_name')}</span><span class="chip">⚡ real-time on CPU</span></div>
</div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ settings (on the page, so phones can reach them)
with st.expander("⚙️ Settings", expanded=False):
    mode_label = st.radio("Input type", ["Face photo (find eyes automatically)", "Eye close-up (use image as is)"],
                          help="The model was trained on small eye crops. For a normal selfie / webcam photo use the first option.")
    mode = "auto" if mode_label.startswith("Face") else "crop"
    sc1, sc2 = st.columns(2)
    thr = sc1.slider("Closed-eye threshold", 0.05, 0.95, float(default_thr), 0.01,
                     help="Probability above which an eye is called CLOSED. Lower = more sensitive (fewer missed closed eyes, more false alarms).")
    alert_n = sc2.slider("Alert after N closed snapshots in a row", 2, 6, 3)
    st.caption(f"🧠 Model: **{cfg.get('model_name')}** - trained threshold {default_thr:.2f}")


def show_result(rgb):
    ph = st.empty()
    ph.markdown('<div class="loader"><div class="spin"></div><p>ANALYSING ...</p></div>', unsafe_allow_html=True)
    crops, boxes = get_crops(rgb, mode)
    probs = predict(model, crops) if crops else []
    time.sleep(0.6)                                   # cosmetic: lets the loading wheel show briefly
    ph.empty()

    if not crops:
        st.warning("No eyes found. Try a clearer, front-facing photo with good light - or switch to **Eye close-up** in the sidebar.")
        st.image(rgb, width="stretch"); return None

    flags = [p >= thr for p in probs]
    p_mean = float(probs.mean()); closed = p_mean >= thr
    tint = "#ef4444" if closed else "#22c55e"

    left, right = st.columns([3, 2], gap="large")
    with left, st.container(border=True):
        st.image(annotate(rgb, boxes, flags) if mode == "auto" else rgb, width="stretch",
                 caption="Green box = open, red box = closed" if mode == "auto" else None)
    with right, st.container(border=True):
        st.markdown(gauge(p_mean, 220, 18, "Closed probability", tint, cls="big"), unsafe_allow_html=True)
        st.markdown(f'<div class="pill {"closed" if closed else "open"}">{"😴 EYES CLOSED" if closed else "👀 EYES OPEN"}</div>'
                    f'<div class="sub">decision threshold {thr:.0%}</div>', unsafe_allow_html=True)
        if len(probs) > 1:
            eyes = "".join(f'<div>{gauge(p, 96, 9, "", "#ef4444" if p >= thr else "#22c55e")}'
                           f'<div class="eyecap">Eye {i + 1}: {"closed" if p >= thr else "open"}</div></div>' for i, p in enumerate(probs))
            st.markdown(f'<div class="eyes">{eyes}</div>', unsafe_allow_html=True)
        with st.expander("What the model sees (64x64 input)"):
            st.image([c[..., 0] for c in crops], width=110, clamp=True)
    return closed


# ------------------------------------------------------------------ tabs
tab_cam, tab_up = st.tabs(["📷  Selfie camera", "📱  Phone camera / upload"])

with tab_cam:
    st.write("Take a snapshot. Take several in a row to test the **drowsiness alert**.")
    st.caption("📱 On a phone this uses the front camera. To use the **back camera**, open the *Phone camera / upload* tab.")
    if "history" not in st.session_state:
        st.session_state.history, st.session_state.last = [], None
    shot = st.camera_input("Look at the camera", label_visibility="collapsed")
    if shot is not None:
        res = show_result(read_image(shot))
        h = hashlib.md5(shot.getvalue()).hexdigest()
        if res is not None and h != st.session_state.last:            # count each snapshot only once
            st.session_state.history.append(bool(res)); st.session_state.last = h
    hist = st.session_state.history
    if hist:
        with st.container(border=True):
            st.markdown("**Snapshot history** (latest on the right)")
            st.markdown('<div class="hist">' + "".join(f"<span>{'😴' if c else '👀'}</span>" for c in hist[-12:]) + "</div>", unsafe_allow_html=True)
            if len(hist) >= alert_n and all(hist[-alert_n:]):
                st.markdown(f'<div class="alert">🚨 DROWSINESS ALERT - eyes closed in the last {alert_n} snapshots in a row!</div>', unsafe_allow_html=True)
            if st.button("Reset history"):
                st.session_state.history, st.session_state.last = [], None; st.rerun()

with tab_up:
    st.info("📱 **On a phone:** tap *Browse files* and choose **Take photo** - your phone camera opens, and you can switch between the "
            "**front and back** camera there. On a computer you can upload any picture.")
    up = st.file_uploader("Take or upload a photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if up is not None:
        show_result(read_image(up))
    else:
        st.info("Upload a face photo (default) or an eye close-up (switch the input type in the sidebar).")

st.divider()
st.caption("📊 See the **Model Results** page (left menu) to compare all models, see where they fail, and test one image on every model.")