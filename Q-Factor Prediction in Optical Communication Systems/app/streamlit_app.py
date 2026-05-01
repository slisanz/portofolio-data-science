"""Streamlit app — Q-Factor Prediction in Optical Communication Systems."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.config import FEATURE_COLS, MODELS_DIR, FIGURES_DIR
from src.data import load_scaler
from src.models.ft_transformer import FTTransformer
from src.models.mlp import MLPRegressor
from src.train import predict_torch
from src.uncertainty import mc_dropout_predict


st.set_page_config(page_title="Q-Factor Predictor", page_icon=":satellite:", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = load_scaler()
    xgb_path = MODELS_DIR / "XGBoost_tuned.joblib"
    if not xgb_path.exists():
        xgb_path = MODELS_DIR / "XGBoost.joblib"
    xgb = joblib.load(xgb_path) if xgb_path.exists() else None

    mlp = None
    mlp_path = MODELS_DIR / "mlp.pt"
    if mlp_path.exists():
        mlp = MLPRegressor(len(FEATURE_COLS))
        mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))

    ftt = None
    ftt_path = MODELS_DIR / "ft_transformer.pt"
    if ftt_path.exists():
        ftt = FTTransformer(len(FEATURE_COLS))
        ftt.load_state_dict(torch.load(ftt_path, map_location="cpu"))

    conformal = None
    cpath = MODELS_DIR / "conformal_q.joblib"
    if cpath.exists():
        conformal = joblib.load(cpath)

    return {"scaler": scaler, "xgb": xgb, "mlp": mlp, "ftt": ftt, "conformal": conformal}


def sidebar_inputs() -> np.ndarray:
    st.sidebar.header("Optical Link Parameters")
    vals = {
        "OSNR": st.sidebar.slider("OSNR (norm)", 0.0, 1.0, 0.6, 0.01),
        "Launch_Power": st.sidebar.slider("Launch Power (norm)", 0.0, 1.0, 0.5, 0.01),
        "Fiber_Length": st.sidebar.slider("Fiber Length (norm)", 0.0, 1.0, 0.4, 0.01),
        "Dispersion": st.sidebar.slider("Dispersion (norm)", 0.0, 1.0, 0.3, 0.01),
        "Nonlinear_Effect": st.sidebar.slider("Nonlinear Effect (norm)", 0.0, 1.0, 0.3, 0.01),
    }
    return np.array([[vals[c] for c in FEATURE_COLS]], dtype=np.float32)


def page_predictor(art):
    st.title("Q-Factor Predictor")
    st.caption("From OSNR to optical signal quality — physics-aware, uncertainty-calibrated.")

    raw = sidebar_inputs()
    X = art["scaler"].transform(raw)

    cols = st.columns(3)
    if art["xgb"] is not None:
        cols[0].metric("XGBoost", f"{float(art['xgb'].predict(X)[0]):.3f} dB")
    if art["mlp"] is not None:
        cols[1].metric("MLP", f"{float(predict_torch(art['mlp'], X)[0]):.3f} dB")
    if art["ftt"] is not None:
        cols[2].metric("FT-Transformer", f"{float(predict_torch(art['ftt'], X)[0]):.3f} dB")

    st.divider()
    st.subheader("Conformal Prediction Interval (95%)")
    if art["xgb"] is not None and art["conformal"] is not None:
        point = float(art["xgb"].predict(X)[0])
        q = float(art["conformal"]["q"])
        lo, hi = point - q, point + q
        st.write(f"**Estimate**: {point:.3f} dB  ·  **Interval**: [{lo:.3f}, {hi:.3f}]  (q = {q:.4f})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines", line=dict(width=12, color="lightblue"), name="95% interval"))
        fig.add_trace(go.Scatter(x=[point], y=[0], mode="markers", marker=dict(size=14, color="navy"), name="point"))
        fig.update_layout(height=180, yaxis=dict(visible=False), xaxis_title="Q-Factor (dB)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models and run notebook 07 to enable conformal interval.")

    st.subheader("Epistemic Uncertainty (MC-Dropout, MLP)")
    if art["mlp"] is not None:
        mean_, std_ = mc_dropout_predict(art["mlp"], X, n_samples=50)
        st.write(f"mean = {float(mean_[0]):.3f} dB  ·  std = {float(std_[0]):.4f}")


def page_what_if(art):
    st.title("What-If Analysis")
    if art["xgb"] is None:
        st.warning("XGBoost model not found. Train notebook 04 first."); return

    feat = st.selectbox("Sweep feature", FEATURE_COLS)
    raw = sidebar_inputs()
    grid = np.linspace(0, 1, 100)
    rep = np.repeat(raw, 100, axis=0)
    rep[:, FEATURE_COLS.index(feat)] = grid
    X = art["scaler"].transform(rep)
    preds = art["xgb"].predict(X)
    df = pd.DataFrame({feat: grid, "Q_Factor": preds})
    fig = px.line(df, x=feat, y="Q_Factor", title=f"Q-Factor response vs {feat}")
    st.plotly_chart(fig, use_container_width=True)


def page_comparison():
    st.title("Model Comparison")
    bench_path = FIGURES_DIR / "benchmark.csv"
    if not bench_path.exists():
        st.info("Run notebook 10 to generate benchmark.csv."); return
    df = pd.read_csv(bench_path).set_index("model").sort_values("RMSE")
    st.dataframe(df.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}", "MAPE": "{:.3f}"}))
    fig = px.bar(df.reset_index(), x="model", y="RMSE", title="Test RMSE per model (lower is better)", color="RMSE")
    st.plotly_chart(fig, use_container_width=True)


def page_explainability():
    st.title("Explainability")
    candidates = [
        ("SHAP summary", FIGURES_DIR / "09_shap_summary.png"),
        ("Permutation importance", FIGURES_DIR / "09_perm_importance.png"),
        ("Partial dependence", FIGURES_DIR / "09_pdp.png"),
    ]
    for caption, path in candidates:
        if path.exists():
            st.image(str(path), caption=caption, use_column_width=True)
        else:
            st.info(f"Run notebook 09 to generate: {path.name}")


def page_about():
    st.title("About")
    st.markdown(
        """
        **Q-Factor Prediction in Optical Communication Systems** demonstrates an end-to-end
        regression pipeline that combines:

        - Tree-based learners (XGBoost, LightGBM, CatBoost) tuned with **Optuna**
        - **FT-Transformer** for tabular deep learning
        - **Physics-Informed Neural Network (PINN)** anchoring predictions to the OSNR–Q relation
        - **MC-Dropout** epistemic uncertainty + **Split Conformal Prediction** for calibrated intervals
        - **Stacking ensemble** with a Ridge meta-learner
        - SHAP / PDP / permutation importance for model interpretation

        Inputs are five normalized optical link parameters; the target is the receiver Q-Factor (dB).
        """
    )


PAGES = {
    "1. Predictor": page_predictor,
    "2. What-If Analysis": page_what_if,
    "3. Model Comparison": page_comparison,
    "4. Explainability": page_explainability,
    "5. About": page_about,
}


def main():
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    art = load_artifacts()
    if page in ("1. Predictor", "2. What-If Analysis"):
        PAGES[page](art)
    else:
        PAGES[page]()


if __name__ == "__main__":
    main()
