"""Streamlit entry point for the pneumonia detection demo.

The home page is the predictor: upload a chest X-ray, get a probability and a Grad-CAM
overlay. Other pages live under `pages/` and are auto-discovered by Streamlit.
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.inference import (
    CLASS_NAMES,
    explain,
    load_model,
    load_threshold,
    predict,
    preprocess_pil,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Ray",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the trained model...")
def _get_model(model_path: str):
    return load_model(model_path)


def _model_available() -> bool:
    return MODEL_PATH.exists()


def main() -> None:
    st.title("Pneumonia Detection from Chest X-Ray")
    st.caption(
        "Upload a pediatric chest X-ray. The model returns a probability of pneumonia "
        "and highlights the regions that drove the prediction."
    )

    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "1. Upload a chest X-ray image (JPEG or PNG).\n"
            "2. Wait a moment for the model to run.\n"
            "3. Review the prediction, confidence, and Grad-CAM overlay.\n\n"
            "For a deep dive into the data and model, use the navigation pages on the left."
        )
        st.divider()
        st.subheader("Disclaimer")
        st.markdown(
            "This is a portfolio project. The model is **not** a medical device and must "
            "not be used for clinical diagnosis. Always consult a qualified physician."
        )

    if not _model_available():
        st.warning(
            "The trained model file `models/best_model.keras` is not present yet. "
            "Run `notebooks/04_transfer_learning.ipynb` first to produce it, "
            "or download it from the link in the README."
        )
        st.stop()

    model = _get_model(str(MODEL_PATH))
    threshold = load_threshold(THRESHOLD_PATH)

    uploaded = st.file_uploader(
        "Chest X-ray image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info("Waiting for an image to analyse.")
        return

    image = Image.open(uploaded)
    display_rgb, model_input = preprocess_pil(image)

    with st.spinner("Running model and computing Grad-CAM..."):
        result = predict(model, model_input, threshold)
        overlay = explain(model, display_rgb, model_input, alpha=0.4)

    pneumonia_prob = result["probability_pneumonia"]
    normal_prob = result["probability_normal"]
    label = result["label"]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Original")
        st.image(display_rgb, caption="Resized to 224x224", use_column_width=True)
    with col_b:
        st.subheader("Grad-CAM overlay")
        st.image(overlay, caption="Warmer regions = stronger contribution", use_column_width=True)

    st.divider()

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Prediction", label)
    metric_col2.metric("Pneumonia probability", f"{pneumonia_prob*100:.1f}%")
    metric_col3.metric("Decision threshold", f"{threshold:.2f}")

    st.subheader("Probability breakdown")
    st.progress(min(max(pneumonia_prob, 0.0), 1.0), text=f"PNEUMONIA: {pneumonia_prob*100:.1f}%")
    st.progress(min(max(normal_prob, 0.0), 1.0), text=f"NORMAL: {normal_prob*100:.1f}%")

    if label == CLASS_NAMES[1]:
        st.error(
            "The model considers this X-ray more likely to show signs of pneumonia. "
            "This is a screening signal only and is not a diagnosis."
        )
    else:
        st.success(
            "The model considers this X-ray more likely to be normal. "
            "Recall on the held-out test set is high but not perfect; do not rely on this output for clinical decisions."
        )


if __name__ == "__main__":
    main()
