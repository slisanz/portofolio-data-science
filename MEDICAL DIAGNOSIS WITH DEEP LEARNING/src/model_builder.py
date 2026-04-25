"""Model architectures for chest X-ray pneumonia classification.

Two builders:
- `build_baseline_cnn` is a small custom CNN. It exists to give the transfer-learning
  model something concrete to be compared against in the evaluation notebook.
- `build_densenet121` wraps a DenseNet121 ImageNet backbone with a fresh classifier head.
  The backbone is frozen by default; the transfer-learning notebook unfreezes the top
  layers in a second stage for fine-tuning.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def _classification_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def build_baseline_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3),
                       dropout_rate: float = 0.3) -> tf.keras.Model:
    """Four-block CNN: Conv-BN-Pool-Dropout, then GAP and a sigmoid head."""
    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_prob")(x)

    model = models.Model(inputs, outputs, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=_classification_metrics(),
    )
    return model


def build_densenet121(input_shape: Tuple[int, int, int] = (224, 224, 3),
                      dropout_rate: float = 0.3,
                      dense_units: int = 128) -> tf.keras.Model:
    """DenseNet121 backbone (ImageNet weights, frozen) plus a small classification head."""
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = False

    inputs = layers.Input(shape=input_shape, name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_prob")(x)

    model = models.Model(inputs, outputs, name="densenet121_pneumonia")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=_classification_metrics(),
    )
    return model


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30,
                        learning_rate: float = 1e-5) -> tf.keras.Model:
    """Unfreeze the last `num_layers` of the DenseNet backbone for fine-tuning.

    Returns the recompiled model with a much smaller learning rate so the pretrained
    weights are not destroyed.
    """
    base = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=_classification_metrics(),
    )
    return model


def standard_callbacks(checkpoint_path: str, patience: int = 5):
    """Callbacks shared by baseline and transfer learning notebooks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=patience, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.3, patience=2, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]
