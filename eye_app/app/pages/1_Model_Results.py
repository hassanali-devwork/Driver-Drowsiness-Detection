"""Model Results page: leaderboard, charts, failure analysis, and a live 'ask every model' demo."""
import altair as alt
import pandas as pd
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (inject_css, hero, gauge, load_csv, IMG_DIR, available_models, load_keras, read_image, get_crops, predict)

st.set_page_config(page_title="Model Results", page_icon="📊", layout="wide")
inject_css()
hero("📊 Model Results", "How every model performed on people it has never seen. Models were ranked on validation F1; the test set was used once at the end.",
     ["🧪 test set", "🔁 subject-wise split", "🎯 closed-eye focus"])

comp = load_csv("phase2_model_comparison.csv")
if comp is None:
    st.error("results/phase2_model_comparison.csv not found. Copy the Phase 2 result CSVs into the `results/` folder.")
    st.stop()


def show_img(name, caption=None):
    p = IMG_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


comp = comp.sort_values("val_f1", ascending=False).reset_index(drop=True)
best = comp.iloc[0]
st.markdown(f"#### 🏆 Best model: **{best.model}**")
cols = st.columns(4, gap="large")
for col, (lab, val) in zip(cols, [("F1 (closed)", best.f1_closed), ("Recall (closed)", best.recall_closed),
                                  ("Precision (closed)", best.precision_closed), ("AUC", best.auc)]):
    with col, st.container(border=True):
        st.markdown(gauge(float(val), 150, 13, lab, "#6366f1"), unsafe_allow_html=True)

t1, t2, t3, t4, t5, t6 = st.tabs(["🏁 Leaderboard", "📈 Charts", "🔍 Where it fails", "🧪 Tuning & CV",
                                  "🤖 Try all models", "📝 Conclusion"])

# ---------- leaderboard ----------
with t1:
    cols = ["model", "family", "accuracy", "precision_closed", "recall_closed", "f1_closed", "auc", "train_time_s", "ms_per_image"]
    st.dataframe(comp[cols].style.format({k: "{:.3f}" for k in cols[2:7]} | {"train_time_s": "{:.0f}", "ms_per_image": "{:.2f}"})
                 .background_gradient(subset=["f1_closed", "auc", "recall_closed"], cmap="Greens"),
                 width="stretch", hide_index=True)
    st.caption("**Recall (closed)** = share of truly closed eyes that were caught - the most important number for drowsiness. "
               "**Precision** = share of 'closed' alarms that were right.")

# ---------- charts ----------
with t2:
    long = comp.melt(id_vars="model", value_vars=["accuracy", "recall_closed", "f1_closed", "auc"], var_name="metric")
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        y=alt.Y("model:N", sort="-x", title=None), x=alt.X("value:Q", scale=alt.Scale(domain=[0.5, 1.0])),
        color="metric:N", yOffset="metric:N", tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")]
    ).properties(height=380, title="Test metrics by model"), width="stretch")
    st.altair_chart(alt.Chart(comp).mark_circle(size=220).encode(
        x=alt.X("ms_per_image:Q", scale=alt.Scale(type="log"), title="ms per image (log scale)"),
        y=alt.Y("f1_closed:Q", scale=alt.Scale(zero=False), title="F1 (closed)"),
        color="model:N", shape="family:N", tooltip=["model", "family", "f1_closed", "ms_per_image"]
    ).properties(height=320, title="Quality vs speed"), width="stretch")
    a, b = st.columns(2)
    with a: show_img("p2_04_roc_pr.png", "ROC and precision-recall curves")
    with b: show_img("p2_05_confusion_matrices.png", "Confusion matrices")
    show_img("p2_02_training_curves.png", "Training curves")

# ---------- failure analysis ----------
with t3:
    bc = load_csv("phase2_best_by_condition.csv")
    if bc is not None:
        bc = bc.assign(label=bc.condition + ": " + bc.group.astype(str))
        st.altair_chart(alt.Chart(bc).mark_bar(color="#e74c3c").encode(
            y=alt.Y("label:N", sort="x", title=None), x=alt.X("recall_closed:Q", scale=alt.Scale(domain=[0, 1]), title="Recall of closed eyes"),
            tooltip=["label", "n", "recall_closed", "accuracy"]).properties(height=380, title=f"{best.model}: recall per recording condition"),
            width="stretch")
        st.dataframe(bc.drop(columns="label"), width="stretch", hide_index=True)
        weak = bc[bc.n >= 100].dropna(subset=["recall_closed"]).sort_values("recall_closed").head(3)
        st.warning("Weakest conditions: " + "; ".join(f"**{r.condition} = {r.group}** (recall {r.recall_closed:.2f}, n={r.n})" for r in weak.itertuples()))
    a, b = st.columns(2)
    with a: show_img("p2_08_error_gallery.png", "Most confident mistakes")
    with b: show_img("p2_09_occlusion.png", "Where the CNN looks (occlusion sensitivity)")
    show_img("p2_07_threshold_tradeoff.png", "Threshold trade-off: precision vs recall")

# ---------- tuning ----------
with t4:
    tune, cv = load_csv("phase2_cnn_tuning.csv"), load_csv("phase2_classical_cv.csv")
    if tune is not None:
        st.subheader("CNN hyper-parameter search (validation AUC)"); st.dataframe(tune.round(4), width="stretch", hide_index=True)
    if cv is not None:
        st.subheader("Classical models: 3-fold cross-validation grouped by subject (AUC)"); st.dataframe(cv.round(4), width="stretch", hide_index=True)

# ---------- try all models ----------
with t5:
    st.write("Give **one image** and see how **every model answers**. Disagreements show where the models differ.")
    models = available_models()
    if not models:
        st.info("No saved model files found in `models/` (expected `p2_*.keras`).")
    else:
        mode_label = st.radio("Input type", ["Face photo (find eyes)", "Eye close-up"], horizontal=True)
        mode = "auto" if mode_label.startswith("Face") else "crop"
        src = st.radio("Source", ["Upload", "Camera"], horizontal=True)
        f = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="cmp_up") if src == "Upload" else st.camera_input("Snapshot", key="cmp_cam")
        if f is not None:
            rgb = read_image(f); crops, _ = get_crops(rgb, mode)
            if not crops:
                st.warning("No eyes found - try an eye close-up.")
            else:
                st.image(rgb, width=320)
                st.caption("What the models actually see (64x64 input):"); st.image([c[..., 0] for c in crops], width=110, clamp=True)
                rows = []
                with st.spinner("Running all models ..."):
                    for name, (path, thr) in models.items():
                        p = float(predict(load_keras(path), crops).mean())
                        rows.append({"model": name, "P(closed)": p, "threshold": thr, "answer": "😴 CLOSED" if p >= thr else "👀 OPEN"})
                df = pd.DataFrame(rows)
                n_closed = (df.answer.str.contains("CLOSED")).sum()
                (st.success if n_closed in (0, len(df)) else st.warning)(
                    f"{n_closed} of {len(df)} models say CLOSED" + ("  - all models agree." if n_closed in (0, len(df)) else "  - the models disagree!"))
                st.dataframe(df.style.format({"P(closed)": "{:.1%}", "threshold": "{:.2f}"}), width="stretch", hide_index=True)
                st.altair_chart(alt.Chart(df).mark_bar().encode(
                    x=alt.X("P(closed):Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("model:N", title=None),
                    color=alt.Color("answer:N", scale=alt.Scale(domain=["😴 CLOSED", "👀 OPEN"], range=["#e74c3c", "#2ecc71"]))
                ).properties(height=200), width="stretch")

# ---------- conclusion ----------
with t6:
    cl = comp[comp.family == "Classical ML"].iloc[0] if (comp.family == "Classical ML").any() else None
    st.markdown(f"""
**What we achieved.** The best model, **{best.model}**, reaches F1 = **{best.f1_closed:.3f}**, recall = **{best.recall_closed:.3f}** and AUC = **{best.auc:.3f}** on people it has never seen.
""" + (f"The best classical model ({cl.model}) reaches F1 = {cl.f1_closed:.3f}, so learning features directly from pixels with a CNN clearly helps." if cl is not None else ""))
    st.markdown("""
**Limitations**
- One image shows the eye state *now*; real drowsiness needs eyes closed for *a period* (PERCLOS / blink duration) - the camera tab therefore uses a multi-snapshot alert.
- Recall drops with **glasses** and **strong reflections** (see *Where it fails*).
- The model was trained on 64x64 infrared-style eye crops; webcam photos in other lighting can behave differently.
- Eye detection uses OpenCV Haar cascades, which can miss eyes in side views or poor light.

**Next steps:** targeted augmentation for glasses / reflections, ensembling, probability calibration, and a temporal model over video frames.
""")
