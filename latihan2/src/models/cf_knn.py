"""Item-kNN dan User-kNN berbasis cosine similarity sparse."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def _topk_sim(sim: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Pertahankan top-k nilai similarity per baris."""
    sim = sim.tolil(copy=False)
    rows, cols, data = [], [], []
    for r in range(sim.shape[0]):
        row_data = np.array(sim.data[r], dtype=np.float32)
        row_cols = np.array(sim.rows[r], dtype=np.int32)
        if row_data.size > k:
            idx = np.argpartition(-row_data, k)[:k]
            row_data = row_data[idx]
            row_cols = row_cols[idx]
        rows.extend([r] * len(row_cols))
        cols.extend(row_cols.tolist())
        data.extend(row_data.tolist())
    return sp.csr_matrix(
        (data, (rows, cols)), shape=sim.shape, dtype=np.float32
    )


class ItemKNN:
    """Item-item cosine kNN. Score user u pada item i = sum sim(i,j)*r_uj / Z."""

    def __init__(self, k: int = 50):
        self.k = k
        self.name = f"ItemKNN(k={k})"

    def fit(self, X: sp.csr_matrix) -> "ItemKNN":
        self.n_items = X.shape[1]
        self.X = X.tocsr()
        # item-item similarity: normalize kolom -> X.T @ X
        X_col_norm = normalize(X, norm="l2", axis=0).tocsc()
        sim = (X_col_norm.T @ X_col_norm).tocsr()
        # nolkan diagonal
        sim.setdiag(0)
        sim.eliminate_zeros()
        # batasi top-k per item (neighbor)
        self.sim = _sparse_topk_rows(sim, self.k)
        return self

    def score_all(self, u: int) -> np.ndarray:
        r_u = self.X[u]  # (1, n_items) sparse
        scores = r_u @ self.sim  # (1, n_items)
        denom = (r_u != 0).astype(np.float32) @ np.abs(self.sim)
        dense = np.asarray(scores.todense()).ravel()
        dd = np.asarray(denom.todense()).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(dd > 1e-9, dense / dd, 0.0)
        return out.astype(np.float32)

    def predict(self, u: int, i: int) -> float:
        all_ = self.score_all(u)
        v = float(all_[i])
        if v == 0.0:
            return 3.5
        return v


class UserKNN:
    """User-user cosine kNN. Dipakai hanya untuk skala kecil (sample user)."""

    def __init__(self, k: int = 50):
        self.k = k
        self.name = f"UserKNN(k={k})"

    def fit(self, X: sp.csr_matrix) -> "UserKNN":
        self.n_items = X.shape[1]
        self.X = X.tocsr()
        X_row_norm = normalize(X, norm="l2", axis=1).tocsr()
        sim = (X_row_norm @ X_row_norm.T).tocsr()
        sim.setdiag(0)
        sim.eliminate_zeros()
        self.sim = _sparse_topk_rows(sim, self.k)
        return self

    def score_all(self, u: int) -> np.ndarray:
        s_u = self.sim[u]       # (1, n_users)
        scores = s_u @ self.X   # (1, n_items)
        denom = np.abs(s_u) @ (self.X != 0).astype(np.float32)
        dense = np.asarray(scores.todense()).ravel()
        dd = np.asarray(denom.todense()).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(dd > 1e-9, dense / dd, 0.0)
        return out.astype(np.float32)

    def predict(self, u: int, i: int) -> float:
        v = float(self.score_all(u)[i])
        return v if v != 0.0 else 3.5


def _sparse_topk_rows(mat: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Top-k per baris versi vektor (lebih cepat dari versi lil)."""
    indptr = mat.indptr
    data = mat.data
    indices = mat.indices
    new_data, new_indices, new_indptr = [], [], [0]
    for r in range(mat.shape[0]):
        start, end = indptr[r], indptr[r + 1]
        row_data = data[start:end]
        row_idx = indices[start:end]
        if row_data.size > k:
            top = np.argpartition(-row_data, k)[:k]
            row_data = row_data[top]
            row_idx = row_idx[top]
        new_data.extend(row_data.tolist())
        new_indices.extend(row_idx.tolist())
        new_indptr.append(len(new_data))
    return sp.csr_matrix(
        (new_data, new_indices, new_indptr), shape=mat.shape, dtype=np.float32
    )
