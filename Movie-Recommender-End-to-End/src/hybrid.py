"""Hybrid weighted: gabungan skor ALS + Content-based dengan normalisasi z-score."""

from __future__ import annotations

import numpy as np


class WeightedHybrid:
    def __init__(self, als_model, content_model, alpha: float = 0.7):
        self.als = als_model
        self.content = content_model
        self.alpha = alpha
        self.n_items = als_model.n_items
        self.name = f"Hybrid(ALS*{alpha:.2f}+Content*{1-alpha:.2f})"

    def fit(self, *args, **kwargs):
        return self

    @staticmethod
    def _z(x: np.ndarray) -> np.ndarray:
        mu = x.mean()
        sd = x.std()
        if sd < 1e-9:
            return x - mu
        return (x - mu) / sd

    def score_all(self, u: int) -> np.ndarray:
        a = self._z(self.als.score_all(u))
        c = self._z(self.content.score_all(u))
        return (self.alpha * a + (1 - self.alpha) * c).astype(np.float32)

    def predict(self, u: int, i: int) -> float:
        return self.als.predict(u, i)
