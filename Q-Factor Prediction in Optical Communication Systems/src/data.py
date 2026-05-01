from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLS,
    PROCESSED_DIR,
    RAW_CSV,
    SEED,
    SPLIT_RATIOS,
    TARGET_COL,
)


def load_raw(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, nrows=nrows)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()


def make_splits(df: pd.DataFrame, seed: int = SEED):
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    df_train, df_tmp = train_test_split(df, test_size=(1 - train_ratio), random_state=seed)
    rel_test = test_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(df_tmp, test_size=rel_test, random_state=seed)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    return scaler


def save_processed(splits: dict, scaler: StandardScaler) -> None:
    for name, (X, y) in splits.items():
        np.savez_compressed(PROCESSED_DIR / f"{name}.npz", X=X, y=y)
    joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")


def load_processed(name: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(PROCESSED_DIR / f"{name}.npz")
    return arr["X"], arr["y"]


def load_scaler() -> StandardScaler:
    return joblib.load(PROCESSED_DIR / "scaler.joblib")


def prepare_and_save(nrows: int | None = None) -> None:
    df = load_raw(nrows=nrows)
    train_df, val_df, test_df = make_splits(df)
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    scaler = fit_scaler(X_train)
    splits = {
        "train": (scaler.transform(X_train.values), y_train.values.astype(np.float32)),
        "val": (scaler.transform(X_val.values), y_val.values.astype(np.float32)),
        "test": (scaler.transform(X_test.values), y_test.values.astype(np.float32)),
    }
    save_processed(splits, scaler)
