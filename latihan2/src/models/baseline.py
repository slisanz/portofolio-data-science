"""Baseline RecSys: global mean, user mean, item mean, popularity."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class GlobalMean:
    name = "GlobalMean"

    def fit(self, X: sp.csr_matrix) -> "GlobalMean":
        data = X.data
        self.mu = float(data.mean()) if data.size else 3.5
        self.n_items = X.shape[1]
        return self

    def predict(self, u: int, i: int) -> float:
        return self.mu

    def score_all(self, u: int) -> np.ndarray:
        return np.full(self.n_items, self.mu, dtype=np.float32)


class UserMean:
    name = "UserMean"

    def fit(self, X: sp.csr_matrix) -> "UserMean":
        self.n_items = X.shape[1]
        self.mu = float(X.data.mean()) if X.data.size else 3.5
        sums = np.asarray(X.sum(axis=1)).ravel()
        counts = np.asarray((X > 0).sum(axis=1)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            self.user_mean = np.where(counts > 0, sums / np.maximum(counts, 1), self.mu)
        return self

    def predict(self, u: int, i: int) -> float:
        return float(self.user_mean[u])

    def score_all(self, u: int) -> np.ndarray:
        return np.full(self.n_items, self.user_mean[u], dtype=np.float32)


class ItemMean:
    name = "ItemMean"

    def fit(self, X: sp.csr_matrix) -> "ItemMean":
        self.mu = float(X.data.mean()) if X.data.size else 3.5
        sums = np.asarray(X.sum(axis=0)).ravel()
        counts = np.asarray((X > 0).sum(axis=0)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            self.item_mean = np.where(counts > 0, sums / np.maximum(counts, 1), self.mu).astype(np.float32)
        return self

    def predict(self, u: int, i: int) -> float:
        return float(self.item_mean[i])

    def score_all(self, u: int) -> np.ndarray:
        return self.item_mean


class Popularity:
    """Score = jumlah interaksi di train (untuk ranking). RMSE pakai mean rating."""

    name = "Popularity"

    def fit(self, X: sp.csr_matrix) -> "Popularity":
        self.mu = float(X.data.mean()) if X.data.size else 3.5
        self.pop = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
        sums = np.asarray(X.sum(axis=0)).ravel()
        counts = np.asarray((X > 0).sum(axis=0)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            self.item_mean = np.where(counts > 0, sums / np.maximum(counts, 1), self.mu).astype(np.float32)
        return self

    def predict(self, u: int, i: int) -> float:
        return float(self.item_mean[i])

    def score_all(self, u: int) -> np.ndarray:
        return self.pop
