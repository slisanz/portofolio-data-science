"""Content-based rekomendasi memakai embedding tag-genome (1128-dim)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class GenomeContentBased:
    """Rekomendasi via cosine similarity di ruang embedding genome.

    Profil user = rata-rata embedding film yang dia rating (dibobot rating).
    Skor item = cosine(user_profile, item_embedding).
    """

    def __init__(self, paths):
        self.paths = paths
        self.name = "ContentBased(genome)"

    def fit(self, X: sp.csr_matrix, item_ids: np.ndarray) -> "GenomeContentBased":
        self.n_items = X.shape[1]
        self.X = X.tocsr()
        genome = pl.read_parquet(self.paths.processed_dir / "genome_embedding.parquet")
        g_movie = genome["movieId"].to_numpy()
        emb = np.asarray(genome["embedding"].to_list(), dtype=np.float32)

        # petakan embedding ke indeks item yang dipakai
        item_to_idx = {int(m): i for i, m in enumerate(item_ids)}
        E = np.zeros((self.n_items, emb.shape[1]), dtype=np.float32)
        mask = np.zeros(self.n_items, dtype=bool)
        for row, mid in enumerate(g_movie):
            j = item_to_idx.get(int(mid))
            if j is not None:
                E[j] = emb[row]
                mask[j] = True
        # normalisasi L2
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.E = E / norms
        self.has_content = mask
        # precompute user profiles (sparse weighted mean of E)
        self.user_profiles = normalize(X @ self.E, norm="l2", axis=1)
        return self

    def score_all(self, u: int) -> np.ndarray:
        profile = self.user_profiles[u]
        scores = self.E @ profile
        scores[~self.has_content] = -1.0
        return scores.astype(np.float32)

    def predict(self, u: int, i: int) -> float:
        return 3.5  # content-based murni ranking
