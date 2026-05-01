from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class StackingRegressor:
    """Out-of-fold-free stacker: fit base learners on train, meta on val predictions."""

    def __init__(self, base_predictions_train: dict, y_train: np.ndarray, alpha: float = 1.0):
        names = sorted(base_predictions_train.keys())
        self.names = names
        X_meta = np.column_stack([base_predictions_train[n] for n in names])
        self.meta = Ridge(alpha=alpha)
        self.meta.fit(X_meta, y_train)

    def predict(self, base_predictions: dict) -> np.ndarray:
        X_meta = np.column_stack([base_predictions[n] for n in self.names])
        return self.meta.predict(X_meta)
