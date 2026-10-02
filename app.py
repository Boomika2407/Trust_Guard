"""TrustGuard: Credit Card Fraud Detector (Streamlit UI).

Run with:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.charts import contribution_figure, distribution_figure, probability_figure
from src.features import build_features, missing_columns, normalise_columns
from src.predictor import FraudPredictor, load_metrics
from src.rules import triage
from src.samples import FRAUD_SAMPLE, LEGIT_SAMPLE, sample_batch_csv

st.set_page_config(page_title="TrustGuard | Credit Card Fraud Detector",
                   page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

if config.STYLE_PATH.exists():
    st.markdown(f"<style>{config.STYLE_PATH.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> FraudPredictor:
    return FraudPredictor()


try:
    predictor = get_predictor()
except Exception as exc:  # missing/incompatible pickle
    st.error(f"Could not load the model: {exc}")
    st.stop()

metrics = load_metrics()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ TrustGuard")
    st.markdown("""<div class="side-card">
    🔹 Algorithm: <b>XGBoost</b><br>
    🔹 Imbalance: <b>scale_pos_weight</b><br>
    🔹 Tuning: <b>Threshold optimisation</b><br>
    🔹 Explainability: <b>SHAP</b><br>
    🔹 Pipeline: <b>scikit-learn Pipeline</b></div>""", unsafe_allow_html=True)

    st.markdown("**Decision threshold**")
    threshold = st.slider("Flag as fraud when probability ≥", 0.01, 0.99,
                          float(metrics["threshold"]), 0.01,
                          help="Lower values catch more fraud but raise more false alarms.")
    use_rules = st.toggle("Apply v14 / v4 triage rules", value=True,
                          help="Auto-block when v14 < -5 and v4 > 5; send to review if only one holds.")

    st.markdown("""<div class="side-note"><br><b>Dataset</b><br>
    <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Kaggle: Credit Card Fraud</a><br>
    Features: Time, V1-V28 (PCA), Amount<br>Target: Class (0 = legit, 1 = fraud)</div>""",
                unsafe_allow_html=True)
    st.caption("For demonstration and academic use only. Not for production financial decisions.")

# ── Header and headline metrics ───────────────────────────────────────────────
st.markdown("""<div class="hero">
  <div class="hero-title">🛡️ Trust<span>Guard</span></div>
  <p class="hero-sub">An end-to-end machine learning system for credit card fraud detection,
  trained on 284,807 real transactions and powered by XGBoost with SHAP explanations.</p>
  <div class="badge-row"><span class="badge">XGBoost</span><span class="badge">ROC-AUC 0.9725</span>
  <span class="badge">284,807 transactions</span><span class="badge">SHAP explainability</span></div>
</div>""", unsafe_allow_html=True)

cards = [(f"{metrics['roc_auc']:.4f}", "ROC-AUC"), (f"{metrics['precision']:.4f}", "Precision"),
         (f"{metrics['recall']:.2f}", "Recall"), (f"{metrics['fraud_rate']:.2f}%", "Fraud rate"),
         (f"{int(metrics['n_transactions']):,}", "Transactions")]
st.markdown('<div class="metric-row">' + "".join(
    f'<div class="metric-card"><div class="metric-val">{v}</div><div class="metric-lbl">{l}</div></div>'
    for v, l in cards) + "</div>", unsafe_allow_html=True)

tab_single, tab_batch, tab_model = st.tabs(
    ["🔢  Single Transaction", "📂  Batch CSV Upload", "📊  Model Insights"])


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_sample(sample: dict, kind: str) -> None:
    """Push a sample into the input widgets' session state."""
    st.session_state["time"] = float(sample["time"])
    st.session_state["amount"] = float(sample["amount"])
    for i, val in enumerate(sample["v"], start=1):
        st.session_state[f"v{i}"] = float(val)
    st.session_state["sample_kind"] = kind


for key in ["time", "amount"] + config.V_COLS:
    st.session_state.setdefault(key, 0.0)

RESULT_STYLE = {
    "fraud": ("result-fraud", "fill-fraud", "⚠️", "FRAUDULENT"),
    "review": ("result-review", "fill-review", "🔎", "NEEDS REVIEW"),
    "legit": ("result-legit", "fill-legit", "✅", "LEGITIMATE"),
}

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: single transaction
# ═════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown('<div class="section-header">Transaction details</div>', unsafe_allow_html=True)
    st.caption("V1-V28 are PCA components from the original Kaggle dataset.")

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    b1.button("⚠️ Load fraud sample", on_click=load_sample, args=(FRAUD_SAMPLE, "fraud"))
    b2.button("✅ Load legit sample", on_click=load_sample, args=(LEGIT_SAMPLE, "legit"))
    b3.button("↺ Reset", on_click=load_sample,
              args=({"time": 0.0, "amount": 0.0, "v": [0.0] * 28}, "none"))

    kind = st.session_state.get("sample_kind")
    if kind == "fraud":
        st.info("Loaded a known fraudulent transaction from the dataset.")
    elif kind == "legit":
        st.success("Loaded a known legitimate transaction from the dataset.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-header">Transaction info</div>', unsafe_allow_html=True)
        st.number_input("Time (seconds since first transaction)", key="time", format="%.2f")
        st.number_input("Amount ($)", key="amount", min_value=0.0, format="%.2f")
        st.markdown('<div class="section-header">V1 - V9</div>', unsafe_allow_html=True)
        for i in range(1, 10):
            st.number_input(f"V{i}", key=f"v{i}", format="%.6f")
    with c2:
        st.markdown('<div class="section-header">V10 - V19</div>', unsafe_allow_html=True)
        for i in range(10, 20):
            st.number_input(f"V{i}", key=f"v{i}", format="%.6f")
    with c3:
        st.markdown('<div class="section-header">V20 - V28</div>', unsafe_allow_html=True)
        for i in range(20, 29):
            st.number_input(f"V{i}", key=f"v{i}", format="%.6f")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    run_col, _ = st.columns([1, 2])
    analyse = run_col.button("🔍  ANALYZE TRANSACTION")

    if analyse:
        raw = pd.DataFrame([{"time": st.session_state["time"], "amount": st.session_state["amount"],
                             **{f"v{i}": st.session_state[f"v{i}"] for i in range(1, 29)}}])
        X = build_features(raw)
        prob = float(predictor.predict_proba(X)[0])
        v14, v4 = float(X.at[0, "v14"]), float(X.at[0, "v4"])
        if use_rules:
            status, action = triage(prob, threshold, v14, v4)
        else:
            status = "fraud" if prob >= threshold else "legit"
            action = ("Block the transaction and flag it for manual review." if status == "fraud"
                      else "Transaction is safe to process.")

        st.markdown('<div class="section-header">Prediction result</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        box, fill, icon, label = RESULT_STYLE[status]
        with left:
            st.markdown(f"""<div class="{box}"><div style="font-size:2.8rem;margin-bottom:8px">{icon}</div>
              <div class="result-label">{label}</div>
              <div class="result-prob">Fraud probability: <b>{prob:.2%}</b> (threshold {threshold:.2f})</div>
              <div class="prob-bar-wrap"><div class="prob-bar-fill {fill}" style="width:{prob * 100:.1f}%"></div></div>
              <div style="font-size:0.75rem;color:#64748b">Risk score</div></div>""",
                        unsafe_allow_html=True)
            {"fraud": st.error, "review": st.warning, "legit": st.success}[status](action)
        with right:
            fig = probability_figure(prob)
            st.pyplot(fig)
            plt.close(fig)
            m1, m2 = st.columns(2)
            m1.metric("Fraud probability", f"{prob:.4f}")
            m2.metric("Confidence", f"{max(prob, 1 - prob):.2%}")

        with st.expander("🧠  Why did the model make this prediction? (SHAP)", expanded=True):
            if not predictor.supports_shap:
                st.info("SHAP explanations need an XGBoost model. The loaded model is a different type.")
            else:
                contribs, base = predictor.contributions(X)
                fig = contribution_figure(contribs.iloc[0], X.iloc[0])
                st.pyplot(fig)
                plt.close(fig)
                top = contribs.iloc[0].abs().sort_values(ascending=False).head(3).index
                parts = [f"`{n}` ({'toward fraud' if contribs.iloc[0][n] > 0 else 'toward legit'})" for n in top]
                st.caption("Largest drivers: " + ", ".join(parts) +
                           ". Values are in log-odds; red bars raise fraud risk.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: batch upload
# ═════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-header">Batch transaction analysis</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV from the Kaggle credit card dataset. Required columns (any order, any case):")
    st.code("Time, V1, V2, ..., V28, Amount", language="text")
    st.caption("An optional `Class` column is kept in the output and used to report accuracy metrics.")
    st.download_button("⬇️ Download a 2-row sample CSV", sample_batch_csv(),
                       file_name="sample_transactions.csv", mime="text/csv")

    uploaded = st.file_uploader("Drop your CSV file here", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = normalise_columns(pd.read_csv(uploaded))
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()

        st.markdown(f"**Preview: {len(df_raw):,} transactions loaded**")
        st.dataframe(df_raw.head(5), width="stretch")
        missing = missing_columns(df_raw)
        if missing:
            st.error(f"Missing columns: {missing}. The CSV needs Time, V1-V28 and Amount.")
        elif st.button("🔍  RUN BATCH PREDICTION"):
            with st.spinner("Scoring transactions..."):
                X = build_features(df_raw)
                probs = predictor.predict_proba(X)
                preds = (probs >= threshold).astype(int)

            res = df_raw.copy()
            res["fraud_probability"] = np.round(probs, 6)
            res["prediction"] = preds
            res["result"] = np.where(preds == 1, "Fraud", "Legitimate")
            n_fraud, n_legit = int(preds.sum()), int((preds == 0).sum())

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Batch results summary</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total", f"{len(preds):,}")
            k2.metric("🚨 Fraudulent", f"{n_fraud:,}")
            k3.metric("✅ Legitimate", f"{n_legit:,}")
            k4.metric("Flag rate", f"{n_fraud / len(preds) * 100:.3f}%")

            if "class" in df_raw.columns:
                y = df_raw["class"].astype(int).to_numpy()
                tp = int(((preds == 1) & (y == 1)).sum())
                fp = int(((preds == 1) & (y == 0)).sum())
                fn = int(((preds == 0) & (y == 1)).sum())
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Caught fraud (TP)", tp)
                a2.metric("Missed fraud (FN)", fn)
                a3.metric("False alarms (FP)", fp)
                a4.metric("Precision / Recall", f"{prec:.2f} / {rec:.2f}")

            fig = distribution_figure(n_legit, n_fraud)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown('<div class="section-header">Transaction-level results</div>', unsafe_allow_html=True)
            show = res.sort_values("fraud_probability", ascending=False)
            cols = [c for c in ["time", "amount", "class", "fraud_probability", "prediction", "result"]
                    if c in show.columns]
            st.dataframe(show[cols], width="stretch", height=380)
            st.download_button("⬇️  Download full results as CSV",
                               res.to_csv(index=False).encode("utf-8"),
                               file_name="fraudshield_results.csv", mime="text/csv")
            if n_fraud:
                st.warning(f"{n_fraud} suspicious transaction(s) detected. Review the top rows above.")
            else:
                st.success("No fraudulent transactions detected in this batch.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: model insights
# ═════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-header">Model comparison (test set)</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(config.MODEL_COMPARISON).set_index("Model"), width="stretch")
    st.caption("XGBoost was selected on ROC-AUC. Accuracy is ignored because a model that always predicts "
               "'legitimate' already scores 99.83%.")

    st.markdown('<div class="section-header">Explainability plots</div>', unsafe_allow_html=True)
    plots = [("SHAP global importance", "shap_bar.png"), ("SHAP beeswarm", "shap_beeswarm.png"),
             ("XGBoost built-in importance", "feature_importance_xgb.png"),
             ("SHAP dependence (v14, v4)", "shap_dependence.png"),
             ("Waterfall: fraud case", "shap_waterfall_fraud.png"),
             ("Waterfall: legitimate case", "shap_waterfall_legit.png")]
    available = [(t, config.PLOTS_DIR / f) for t, f in plots if (config.PLOTS_DIR / f).exists()]
    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, available[i:i + 2]):
            col.image(str(path), caption=title, width="stretch")

    st.markdown('<div class="section-header">Key findings</div>', unsafe_allow_html=True)
    st.markdown("""
- **v14 dominates.** Values below -5 give SHAP contributions of +4 to +8, close to certain fraud.
- **v4 scales risk.** Higher values raise fraud probability steadily.
- **Two-stage alerting.** Auto-block when `v14 < -5` and `v4 > 5`; manual review when only one holds.
- **No resampling.** `scale_pos_weight` handles the imbalance, so there is no SMOTE leakage risk.
- **Retrain monthly.** Track fraud-class recall and retrain if it falls below 0.85.
""")

st.markdown("""<div class="footer">TrustGuard &nbsp;|&nbsp; XGBoost + SHAP + Streamlit</div>""", unsafe_allow_html=True)
