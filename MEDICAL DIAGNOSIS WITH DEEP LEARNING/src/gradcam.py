"""Grad-CAM for the DenseNet121 pneumonia classifier.

Grad-CAM (Selvaraju et al., 2017) produces a class-discriminative heatmap by weighting the
final convolutional feature maps by the gradient of the predicted class with respect to
those maps. For a binary sigmoid output we treat the single logit as the target.

For DenseNet121 the standard target layer is the last convolutional block:
`conv5_block16_concat`. The functions here are model-agnostic; the layer name is just the
default.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf

DEFAULT_LAST_CONV = "conv5_block16_concat"


def _find_last_conv_layer(model: tf.keras.Model, preferred: str = DEFAULT_LAST_CONV) -> tf.keras.layers.Layer:
    """Find the target conv layer, even when the backbone is wrapped inside a parent model."""
    for layer in model.layers:
        if layer.name == preferred:
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return layer.get_layer(preferred)
            except ValueError:
                continue
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No convolutional layer found for Grad-CAM.")


def _build_gradcam_model(model: tf.keras.Model, target_layer: tf.keras.layers.Layer
                        ) -> tf.keras.Model:
    """Submodel that returns (target_layer_activations, predictions).

    When the target layer lives inside a wrapped backbone (e.g. DenseNet121),
    Keras 3 forbids referencing its `.output` from the parent graph. We rebuild
    the graph: input -> backbone (exposing both target activation and backbone
    output) -> remaining head layers.
    """
    if target_layer in model.layers:
        return tf.keras.Model(model.inputs, [target_layer.output, model.output])

    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                if layer.get_layer(target_layer.name) is target_layer:
                    backbone = layer
                    break
            except ValueError:
                continue
    if backbone is None:
        raise ValueError(f"Could not locate backbone owning '{target_layer.name}'.")

    inner = tf.keras.Model(backbone.inputs, [target_layer.output, backbone.output])

    new_input = tf.keras.Input(shape=model.input_shape[1:])
    target_act, x = inner(new_input)
    seen_backbone = False
    for layer in model.layers:
        if layer is backbone:
            seen_backbone = True
            continue
        if seen_backbone and not isinstance(layer, tf.keras.layers.InputLayer):
            x = layer(x)

    return tf.keras.Model(new_input, [target_act, x])


def make_gradcam_heatmap(image: np.ndarray, model: tf.keras.Model,
                         last_conv_layer_name: str = DEFAULT_LAST_CONV) -> np.ndarray:
    """Return a 2D heatmap (values 0..1) the same height/width as the conv feature map.

    `image` must already be preprocessed for the model (shape (H, W, 3), float32).
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    target_layer = _find_last_conv_layer(model, last_conv_layer_name)
    grad_model = _build_gradcam_model(model, target_layer)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image, training=False)
        target = predictions[:, 0]

    grads = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / max_val if max_val > 0 else heatmap
    return heatmap.numpy()


def overlay_heatmap(original_uint8: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Resize the heatmap to the original image and blend it as a JET overlay.

    `original_uint8` should be RGB uint8 with shape (H, W, 3).
    """
    h, w = original_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    coloured = cv2.applyColorMap(heatmap_uint8, colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(original_uint8, 1 - alpha, coloured, alpha, 0)
    return blended


def gradcam_pipeline(original_uint8: np.ndarray, preprocessed: np.ndarray,
                     model: tf.keras.Model,
                     last_conv_layer_name: str = DEFAULT_LAST_CONV,
                     alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """End-to-end helper: returns (raw_heatmap, overlay_rgb)."""
    heatmap = make_gradcam_heatmap(preprocessed, model, last_conv_layer_name)
    overlay = overlay_heatmap(original_uint8, heatmap, alpha=alpha)
    return heatmap, overlay
