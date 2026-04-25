"""Model performance page.

All numbers are read from `reports/`. Re-run the evaluation notebook to refresh them.
"""

from pathlib import Path

import json
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance")
st.caption("Metrics computed on the held-out test set after threshold tuning.")


@st.cache_data
def load_metrics():
    p = REPORTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_comparison():
    p = REPORTS_DIR / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_classification_report():
    p = REPORTS_DIR / "classification_report.txt"
    return p.read_text() if p.exists() else None


metrics = load_metrics()
comparison = load_comparison()

if metrics is None:
    st.info("Run `notebooks/05_evaluation_gradcam.ipynb` to populate the metrics file.")
    st.stop()

dn = metrics["densenet_test"]
bl = metrics["baseline_test"]
threshold = metrics["chosen_threshold"]

st.subheader("DenseNet121 at the chosen threshold")
a, b, c, d, e = st.columns(5)
a.metric("Accuracy", f"{dn['accuracy']*100:.2f}%")
b.metric("Precision", f"{dn['precision']*100:.2f}%")
c.metric("Recall", f"{dn['recall']*100:.2f}%")
d.metric("F1", f"{dn['f1']*100:.2f}%")
e.metric("AUC", f"{dn['auc']:.3f}")
st.caption(f"Decision threshold = {threshold:.2f}. Recall is the main objective in this medical context.")

st.subheader("Baseline vs DenseNet121")
if comparison is not None:
    st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
else:
    st.info("Run the evaluation notebook to generate `model_comparison.csv`.")

st.subheader("Confusion matrix")
cm_path = FIG_DIR / "confusion_matrix.png"
if cm_path.exists():
    st.image(str(cm_path), use_column_width=True)

st.subheader("ROC curve")
roc_path = FIG_DIR / "roc_curve.png"
if roc_path.exists():
    st.image(str(roc_path), use_column_width=True)

st.subheader("Precision-Recall curve")
pr_path = FIG_DIR / "pr_curve.png"
if pr_path.exists():
    st.image(str(pr_path), use_column_width=True)

st.subheader("Threshold sweep")
ts_path = FIG_DIR / "threshold_tuning.png"
if ts_path.exists():
    st.image(str(ts_path), use_column_width=True)
    st.caption(
        "Each point is a candidate threshold. The chosen value is the smallest threshold "
        "that keeps recall on PNEUMONIA at or above 0.95, with the best precision among the qualifying candidates."
    )

st.subheader("Grad-CAM samples")
gc_path = FIG_DIR / "gradcam_samples.png"
if gc_path.exists():
    st.image(str(gc_path), use_column_width=True)
    st.caption(
        "True positives, true negatives, and false negatives. The false negatives are "
        "particularly informative because they show where the model is still missing pneumonia signs."
    )

st.subheader("Classification report")
report = load_classification_report()
if report is not None:
    st.code(report, language="text")
