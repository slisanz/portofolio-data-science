from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLS


def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_COLS].copy()
    eps = 1e-6
    out["osnr_over_nl"] = out["OSNR"] / (out["Nonlinear_Effect"] + eps)
    out["power_x_length"] = out["Launch_Power"] * out["Fiber_Length"]
    out["disp_x_length"] = out["Dispersion"] * out["Fiber_Length"]
    out["nl_x_power"] = out["Nonlinear_Effect"] * out["Launch_Power"]
    out["log_osnr"] = np.log1p(out["OSNR"])
    out["log_power"] = np.log1p(out["Launch_Power"])
    return out


def polynomial_pairs(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or FEATURE_COLS
    out = df.copy()
    for i, a in enumerate(cols):
        for b in cols[i:]:
            out[f"{a}_x_{b}"] = df[a] * df[b]
    return out
