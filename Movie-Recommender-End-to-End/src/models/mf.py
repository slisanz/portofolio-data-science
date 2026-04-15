"""Matrix factorization: SVD (scipy), ALS implicit, BPR-MF.

Catatan: `scikit-surprise` tidak bisa dibuild di Windows tanpa MSVC sehingga
SVD direalisasikan via `scipy.sparse.linalg.svds` pada matrix centered. Ini
ekuivalen secara matematis dengan FunkSVD tanpa regularisasi dan lebih cepat
untuk matriks sparse besar yang kita pakai.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class SVDModel:
    def __init__(self, k: int = 64):
        self.k = k
        self.name = f"SVD(k={k})"

    def fit(self, X: sp.csr_matrix) -> "SVDModel":
        self.n_items = X.shape[1]
        self.mu = float(X.data.mean()) if X.data.size else 3.5
        counts_u = np.asarray((X > 0).sum(axis=1)).ravel()
        counts_i = np.asarray((X > 0).sum(axis=0)).ravel()
        sums_u = np.asarray(X.sum(axis=1)).ravel()
        sums_i = np.asarray(X.sum(axis=0)).ravel()
        self.bu = np.where(counts_u > 0, sums_u / np.maximum(counts_u, 1) - self.mu, 0.0).astype(np.float32)
        self.bi = np.where(counts_i > 0, sums_i / np.maximum(counts_i, 1) - self.mu, 0.0).astype(np.float32)

        # Center the nonzeros agar SVD fokus ke residu
        Xc = X.copy().astype(np.float32)
        rows, cols = Xc.nonzero()
        Xc.data = Xc.data - self.mu - self.bu[rows] - self.bi[cols]

        U, s, Vt = spla.svds(Xc, k=self.k)
        # svds returns ascending singular values
        order = np.argsort(-s)
        s = s[order]
        U = U[:, order]
        Vt = Vt[order, :]
        self.U = (U * np.sqrt(s)).astype(np.float32)
        self.V = (Vt.T * np.sqrt(s)).astype(np.float32)
        return self

    def predict(self, u: int, i: int) -> float:
        return float(self.mu + self.bu[u] + self.bi[i] + self.U[u] @ self.V[i])

    def score_all(self, u: int) -> np.ndarray:
        return (self.mu + self.bu[u] + self.bi + self.V @ self.U[u]).astype(np.float32)


class ALSModel:
    def __init__(self, factors: int = 64, iterations: int = 15, regularization: float = 0.05):
        import implicit  # noqa: F401
        self.factors = factors
        self.iterations = iterations
        self.reg = regularization
        self.name = f"ALS(f={factors},it={iterations})"

    def fit(self, X: sp.csr_matrix) -> "ALSModel":
        import implicit
        # implicit mengharapkan confidence matrix (user x item). Kita pakai
        # rating sebagai confidence proxy (>=0).
        self.n_items = X.shape[1]
        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.reg,
            use_gpu=False,
            calculate_training_loss=False,
        )
        model.fit(X)
        self.model = model
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        return self

    def predict(self, u: int, i: int) -> float:
        return float(self.user_factors[u] @ self.item_factors[i])

    def score_all(self, u: int) -> np.ndarray:
        return (self.item_factors @ self.user_factors[u]).astype(np.float32)


class BPRModel:
    def __init__(self, factors: int = 64, iterations: int = 60, learning_rate: float = 0.01):
        import implicit  # noqa: F401
        self.factors = factors
        self.iterations = iterations
        self.lr = learning_rate
        self.name = f"BPR(f={factors},it={iterations})"

    def fit(self, X: sp.csr_matrix) -> "BPRModel":
        import implicit
        self.n_items = X.shape[1]
        # binarisasi sebagai implicit feedback
        Xb = (X > 0).astype(np.float32)
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            iterations=self.iterations,
            learning_rate=self.lr,
            use_gpu=False,
        )
        model.fit(Xb)
        self.model = model
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        return self

    def predict(self, u: int, i: int) -> float:
        # BPR tidak menghasilkan rating; untuk RMSE pakai fallback mean
        return 3.5

    def score_all(self, u: int) -> np.ndarray:
        return (self.item_factors @ self.user_factors[u]).astype(np.float32)
