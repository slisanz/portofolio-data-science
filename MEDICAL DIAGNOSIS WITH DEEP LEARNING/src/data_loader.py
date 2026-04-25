"""Data pipeline for chest X-ray pneumonia classification.

Handles three concerns:
1. Resplitting the validation set, since the original `val/` folder only contains 16 images.
2. Building a `tf.data.Dataset` pipeline (resize, normalize, augment, batch, prefetch).
3. Computing class weights to counter the train-set imbalance during loss calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def collect_manifest(data_dir: Path) -> pd.DataFrame:
    """Walk the dataset folders and return a dataframe of every image."""
    rows = []
    for split in ("train", "val", "test"):
        for label in CLASS_NAMES:
            folder = Path(data_dir) / split / label
            if not folder.exists():
                continue
            for fp in folder.glob("*.jpeg"):
                rows.append({
                    "filepath": str(fp),
                    "split": split,
                    "label": label,
                    "label_idx": CLASS_TO_INDEX[label],
                })
    return pd.DataFrame(rows)


def resplit_validation(manifest: pd.DataFrame, val_fraction: float = 0.10,
                       seed: int = 42) -> pd.DataFrame:
    """Promote 10 percent of train to val so we get a statistically reliable val set.

    The original `val/` images (16 of them) are kept and merged into the new val set.
    Returns a new manifest with the `split` column reassigned.
    """
    train = manifest[manifest["split"] == "train"].copy()
    val_original = manifest[manifest["split"] == "val"].copy()
    test = manifest[manifest["split"] == "test"].copy()

    train_keep, val_new = train_test_split(
        train, test_size=val_fraction, stratify=train["label"], random_state=seed,
    )
    val_new = val_new.assign(split="val")
    val_combined = pd.concat([val_original, val_new], ignore_index=True)
    train_keep = train_keep.assign(split="train")

    return pd.concat([train_keep, val_combined, test], ignore_index=True)


def compute_train_class_weights(manifest: pd.DataFrame) -> dict:
    """Inverse-frequency class weights for the train split."""
    train_labels = manifest.loc[manifest["split"] == "train", "label_idx"].to_numpy()
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _decode_and_preprocess(filepath: tf.Tensor, label: tf.Tensor):
    """Read JPEG, force 3-channel, resize, apply DenseNet preprocessing."""
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.densenet.preprocess_input(image)
    return image, label


def _build_augmentation() -> tf.keras.Sequential:
    """Light augmentation appropriate for medical X-ray (no vertical flip, no large rotations)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmentation")


def _make_dataset(filepaths, labels, training: bool, batch_size: int,
                  augmentation: tf.keras.Sequential | None) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(2048, len(filepaths)), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def build_datasets(data_dir: str | Path, batch_size: int = BATCH_SIZE,
                   val_fraction: float = 0.10, seed: int = 42
                   ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict, pd.DataFrame]:
    """End-to-end pipeline. Returns (train_ds, val_ds, test_ds, class_weights, manifest)."""
    manifest = collect_manifest(Path(data_dir))
    if manifest.empty:
        raise FileNotFoundError(f"No images found under {data_dir}")
    manifest = resplit_validation(manifest, val_fraction=val_fraction, seed=seed)
    class_weights = compute_train_class_weights(manifest)

    augmentation = _build_augmentation()
    splits = {}
    for split in ("train", "val", "test"):
        sub = manifest[manifest["split"] == split].sample(frac=1, random_state=seed).reset_index(drop=True)
        splits[split] = _make_dataset(
            sub["filepath"].to_numpy(),
            sub["label_idx"].to_numpy().astype(np.float32),
            training=(split == "train"),
            batch_size=batch_size,
            augmentation=augmentation if split == "train" else None,
        )

    return splits["train"], splits["val"], splits["test"], class_weights, manifest
