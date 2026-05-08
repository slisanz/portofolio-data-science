from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED


@dataclass
class KMeansResult:
    model: KMeans
    scaler: StandardScaler
    labels: np.ndarray
    silhouette: float
    k: int


def fit_kmeans(rfm: pd.DataFrame, k_range: Iterable[int] = range(3, 7)) -> KMeansResult:
    feats = rfm[["Recency", "Frequency", "Monetary"]].astype(float).copy()
    feats["Frequency"] = np.log1p(feats["Frequency"])
    feats["Monetary"] = np.log1p(feats["Monetary"])
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(X)
        sil = silhouette_score(X, km.labels_)
        if best is None or sil > best.silhouette:
            best = KMeansResult(model=km, scaler=scaler, labels=km.labels_, silhouette=sil, k=k)
    return best


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def walk_forward_split(series: pd.Series, n_splits: int = 4, horizon: int = 7):
    n = len(series)
    splits = []
    for i in range(n_splits, 0, -1):
        end_train = n - i * horizon
        if end_train <= 30:
            continue
        splits.append((series.iloc[:end_train], series.iloc[end_train : end_train + horizon]))
    return splits
