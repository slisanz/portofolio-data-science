"""Uncertainty quantification utilities.

- MC-Dropout: keep dropout active at inference; multiple stochastic forward passes
  yield epistemic mean + std.
- Split Conformal Prediction: distribution-free prediction intervals with
  finite-sample marginal coverage 1 - alpha (Vovk et al.; Lei et al. 2018).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    X: np.ndarray,
    n_samples: int = 100,
    batch_size: int = 8192,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()  # keep dropout layers stochastic
    preds = []
    for _ in range(n_samples):
        chunk_preds = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
            chunk_preds.append(model(xb).cpu().numpy())
        preds.append(np.concatenate(chunk_preds))
    arr = np.stack(preds, axis=0)  # (n_samples, n_obs)
    return arr.mean(axis=0), arr.std(axis=0)


def split_conformal_intervals(
    y_calib_true: np.ndarray,
    y_calib_pred: np.ndarray,
    y_test_pred: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (lower, upper, q) intervals for y_test_pred using calibration residuals."""
    residuals = np.abs(y_calib_true - y_calib_pred)
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    q = np.sort(residuals)[k]
    return y_test_pred - q, y_test_pred + q, float(q)


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
