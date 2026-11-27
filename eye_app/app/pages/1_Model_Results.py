"""Model Results page: leaderboard, charts, failure analysis, and a live 'ask every model' demo."""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    IMG_DIR,
    available_models,
    gauge,
    get_crops,
    get_theme,
    hero,
    inject_css,
    load_csv,
    load_keras,
    predict,
    read_image,
    render_footer,
    render_navbar,
    render_table,
    section_header,
    style_altair,
)

st.set_page_config(page_title="DrowsAI | Model Results", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_navbar("results")
t = get_theme()

hero(
    title="Model evaluation on unseen drivers.",
    subtitle="How every classical and deep learning model performed on held-out subjects. Ranked on validation F1; evaluated once on the final test split.",
    chips=["Held-Out Test Set", "Subject-Wise Split", "Closed-Eye Recall Focus"],
    eyebrow="BENCHMARK & DIAGNOSTICS",
)

comp = load_csv("phase2_model_comparison.csv")
if comp is None:
    st.error("results/phase2_model_comparison.csv not found. Copy the Phase 2 result CSVs into the results/ folder.")
    st.stop()


def show_img(name: str, caption: str | None = None):
    p = IMG_DIR / name
    if p.exists():
        with st.container(border=True):
            st.image(str(p), caption=caption, width="stretch")


comp = comp.sort_values("val_f1", ascending=False).reset_index(drop=True)
best = comp.iloc[0]

section_header(
    eyebrow="SELECTED CHAMPION",
    title=f"Best model: {best.model}",
    subtitle="Test-set metrics for the closed-eye positive class at the validation-tuned decision threshold.",
)

cols = st.columns(4, gap="medium")
for col, (lab, val) in zip(
    cols,
    [
        ("F1 (closed)", best.f1_closed),
        ("Recall (closed)", best.recall_closed),
        ("Precision (closed)", best.precision_closed),
        ("ROC-AUC", best.auc),
    ],
):
    with col, st.container(border=True):
        st.markdown(gauge(float(val), 154, 13, lab, t["accent"]), unsafe_allow_html=True)

st.write("")
t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Leaderboard", "Charts", "Where It Fails", "Tuning & CV", "Try All Models", "Conclusion"]
)

# ---------- leaderboard ----------
with t1:
    show_cols = [
        "model",
        "family",
        "accuracy",
        "precision_closed",
        "recall_closed",
        "f1_closed",
        "auc",
        "train_time_s",
        "ms_per_image",
    ]
    fmt = {k: "{:.3f}" for k in show_cols[2:7]} | {"train_time_s": "{:.0f}", "ms_per_image": "{:.2f}"}
    styled = comp[show_cols].style.format(fmt).background_gradient(
        subset=["f1_closed", "auc", "recall_closed"], cmap=t["table_cmap"]
    )
    render_table(styled)
    st.caption(
        "Recall (closed) is the share of truly closed eyes caught—the critical metric for drowsiness safety. "
        "Precision is the share of closed alarms that were accurate."
    )

# ---------- charts ----------
with t2:
    long = comp.melt(id_vars="model", value_vars=["accuracy", "recall_closed", "f1_closed", "auc"], var_name="metric")
    chart_metrics = (
        alt.Chart(long)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("model:N", sort="-x", title=None),
            x=alt.X("value:Q", scale=alt.Scale(domain=[0.5, 1.0]), title="Score"),
            color=alt.Color("metric:N", scale=alt.Scale(range=t["chart_palette"])),
            yOffset="metric:N",
            tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=380, title="Test Metrics by Model")
    )
    st.altair_chart(style_altair(chart_metrics), width="stretch")

    chart_speed = (
        alt.Chart(comp)
        .mark_circle(size=220, opacity=0.9)
        .encode(
            x=alt.X("ms_per_image:Q", scale=alt.Scale(type="log"), title="ms per image (log scale)"),
            y=alt.Y("f1_closed:Q", scale=alt.Scale(zero=False), title="F1 (closed)"),
            color=alt.Color("model:N", scale=alt.Scale(range=t["chart_palette"] + ["#94a3b8", "#e29578", "#64748b"])),
            shape="family:N",
            tooltip=["model", "family", alt.Tooltip("f1_closed:Q", format=".3f"), alt.Tooltip("ms_per_image:Q", format=".2f")],
        )
        .properties(height=320, title="Quality vs Inference Speed")
    )
    st.altair_chart(style_altair(chart_speed), width="stretch")

    a, b = st.columns(2, gap="medium")
    with a:
        show_img("p2_04_roc_pr.png", "ROC and precision-recall curves")
    with b:
        show_img("p2_05_confusion_matrices.png", "Confusion matrices")
    show_img("p2_02_training_curves.png", "Training curves")

# ---------- failure analysis ----------
with t3:
    bc = load_csv("phase2_best_by_condition.csv")
    if bc is not None:
        bc = bc.assign(label=bc.condition + ": " + bc.group.astype(str))
        chart_cond = (
            alt.Chart(bc)
            .mark_bar(color=t["danger"], cornerRadiusEnd=4)
            .encode(
                y=alt.Y("label:N", sort="x", title=None),
                x=alt.X("recall_closed:Q", scale=alt.Scale(domain=[0, 1]), title="Recall of closed eyes"),
                tooltip=["label", "n", alt.Tooltip("recall_closed:Q", format=".3f"), alt.Tooltip("accuracy:Q", format=".3f")],
            )
            .properties(height=380, title=f"{best.model}: Recall per Recording Condition")
        )
        st.altair_chart(style_altair(chart_cond), width="stretch")
        render_table(bc.drop(columns="label"))
        weak = bc[bc.n >= 100].dropna(subset=["recall_closed"]).sort_values("recall_closed").head(3)
        st.warning(
            "Weakest conditions: "
            + "; ".join(f"{r.condition} = {r.group} (recall {r.recall_closed:.2f}, n={r.n})" for r in weak.itertuples())
        )
    a, b = st.columns(2, gap="medium")
    with a:
        show_img("p2_08_error_gallery.png", "Most confident mistakes")
    with b:
        show_img("p2_09_occlusion.png", "Where the CNN looks (occlusion sensitivity)")
    show_img("p2_07_threshold_tradeoff.png", "Threshold trade-off: precision vs recall")

# ---------- tuning ----------
with t4:
    tune, cv = load_csv("phase2_cnn_tuning.csv"), load_csv("phase2_classical_cv.csv")
    if tune is not None:
        st.subheader("CNN Hyper-parameter Search (Validation AUC)")
        render_table(tune.round(4))
    if cv is not None:
        st.subheader("Classical Models: 3-Fold Cross-Validation Grouped by Subject (AUC)")
        render_table(cv.round(4))

# ---------- try all models ----------
with t5:
    st.write("Provide one image and compare how every trained deep model responds. Disagreements highlight edge-case ambiguity.")
    models = available_models()
    if not models:
        st.info("No saved model files found in models/ (expected p2_*.keras).")
    else:
        c_opt1, c_opt2 = st.columns(2)
        mode_label = c_opt1.radio("Input type", ["Face photo (find eyes)", "Eye close-up"], horizontal=True)
        mode = "auto" if mode_label.startswith("Face") else "crop"
        src = c_opt2.radio("Source", ["Upload", "Camera"], horizontal=True)
        f = (
            st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="cmp_up")
            if src == "Upload"
            else st.camera_input("Snapshot", key="cmp_cam")
        )
        if f is not None:
            rgb = read_image(f)
            crops, _ = get_crops(rgb, mode)
            if not crops:
                st.warning("No eyes found—try an Eye close-up.")
            else:
                p_left, p_right = st.columns([1, 1], gap="medium")
                with p_left, st.container(border=True):
                    st.image(rgb, width=320)
                with p_right, st.container(border=True):
                    st.caption("Normalized 64x64 model inputs:")
                    st.image([c[..., 0] for c in crops], width=110, clamp=True)

                rows = []
                with st.spinner("Running inference across all models..."):
                    for name, (path, thr) in models.items():
                        p = float(predict(load_keras(path), crops).mean())
                        rows.append(
                            {
                                "model": name,
                                "P(closed)": p,
                                "threshold": thr,
                                "answer": "CLOSED" if p >= thr else "OPEN",
                            }
                        )
                df = pd.DataFrame(rows)
                n_closed = int((df.answer == "CLOSED").sum())
                consensus = n_closed in (0, len(df))
                (st.success if consensus else st.warning)(
                    f"{n_closed} of {len(df)} models predict CLOSED"
                    + (" — unanimous agreement." if consensus else " — models disagree on this frame.")
                )
                render_table(df.style.format({"P(closed)": "{:.1%}", "threshold": "{:.2f}"}))
                chart_cmp = (
                    alt.Chart(df)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("P(closed):Q", scale=alt.Scale(domain=[0, 1]), title="Closed Probability"),
                        y=alt.Y("model:N", title=None),
                        color=alt.Color(
                            "answer:N",
                            scale=alt.Scale(domain=["CLOSED", "OPEN"], range=[t["danger"], t["good"]]),
                        ),
                        tooltip=["model", alt.Tooltip("P(closed):Q", format=".1%"), "threshold", "answer"],
                    )
                    .properties(height=220, title="Per-Model Closed Probability")
                )
                st.altair_chart(style_altair(chart_cmp), width="stretch")

# ---------- conclusion ----------
with t6:
    cl = comp[comp.family == "Classical ML"].iloc[0] if (comp.family == "Classical ML").any() else None
    classical_note = (
        f" By comparison, the strongest classical baseline ({cl.model}) achieves F1 = {cl.f1_closed:.3f}, "
        "confirming that end-to-end pixel representations yield substantial gains."
        if cl is not None
        else ""
    )
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.markdown(
            f"""
            <div class="calm-card">
              <div class="sec-eyebrow">OUTCOME</div>
              <div class="card-h" style="font-size:1.18rem;margin-bottom:10px;">What We Achieved</div>
              <p class="card-p" style="font-size:0.92rem;line-height:1.68;">
                The selected model, <strong>{best.model}</strong>, achieves
                <strong>F1 = {best.f1_closed:.3f}</strong>,
                <strong>Recall = {best.recall_closed:.3f}</strong>, and
                <strong>AUC = {best.auc:.3f}</strong> on drivers completely excluded from training.{classical_note}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="calm-card">
              <div class="sec-eyebrow">ROADMAP</div>
              <div class="card-h" style="font-size:1.18rem;margin-bottom:10px;">Limitations &amp; Next Steps</div>
              <ul style="margin:0;padding-left:18px;color:var(--text-secondary);font-size:0.89rem;line-height:1.68;">
                <li>Single frames classify instantaneous eye state; real fatigue tracking relies on sustained closure (PERCLOS / duration).</li>
                <li>Closed-eye recall decreases with eyewear and strong specular reflections.</li>
                <li>Trained on 64x64 infrared eye crops; visible-light webcams can shift feature contrast.</li>
                <li>Next: glare/glasses augmentation, ensemble calibration, and temporal sequence modeling.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

render_footer()
