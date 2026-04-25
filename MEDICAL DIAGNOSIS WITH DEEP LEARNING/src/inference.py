"""Inference helpers used by the Streamlit app.

Keeps the app code free of TensorFlow specifics and provides a single place to change
preprocessing, model loading, or threshold logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from .gradcam import gradcam_pipeline, DEFAULT_LAST_CONV

IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.5
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def load_model(model_path: str | Path) -> tf.keras.Model:
    """Load the trained Keras model. Wrap in @st.cache_resource at the call site."""
    return tf.keras.models.load_model(str(model_path), compile=False)


def load_threshold(threshold_path: str | Path, default: float = DEFAULT_THRESHOLD) -> float:
    """Read the tuned decision threshold; fall back to 0.5 if file is missing."""
    p = Path(threshold_path)
    if not p.exists():
        return default
    try:
        return float(json.loads(p.read_text())["threshold"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return default


def preprocess_pil(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return (display_rgb_uint8, model_ready_float32). Both are 224x224x3."""
    rgb = image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    display = np.asarray(rgb)
    model_input = tf.keras.applications.densenet.preprocess_input(display.astype(np.float32))
    return display, model_input


def predict(model: tf.keras.Model, model_input: np.ndarray, threshold: float) -> dict:
    """Run prediction and return label, probability, and decision against threshold."""
    batched = np.expand_dims(model_input, axis=0)
    prob = float(model.predict(batched, verbose=0)[0, 0])
    label = CLASS_NAMES[1] if prob >= threshold else CLASS_NAMES[0]
    return {
        "probability_pneumonia": prob,
        "probability_normal": 1.0 - prob,
        "label": label,
        "threshold": threshold,
    }


def explain(model: tf.keras.Model, display_rgb: np.ndarray, model_input: np.ndarray,
            alpha: float = 0.4) -> np.ndarray:
    """Return the Grad-CAM overlay (RGB uint8) for the given preprocessed image."""
    _, overlay = gradcam_pipeline(display_rgb, model_input, model, DEFAULT_LAST_CONV, alpha=alpha)
    return overlay
