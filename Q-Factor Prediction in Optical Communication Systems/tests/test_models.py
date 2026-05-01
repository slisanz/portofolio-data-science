import numpy as np
import torch

from src.models.mlp import MLPRegressor
from src.models.ft_transformer import FTTransformer
from src.models.pinn import PINNRegressor, physics_anchor_q, pinn_loss
from src.evaluate import regression_metrics
from src.uncertainty import split_conformal_intervals, coverage


def test_mlp_forward_shape():
    m = MLPRegressor(5)
    out = m(torch.randn(8, 5))
    assert out.shape == (8,)


def test_ft_transformer_forward_shape():
    m = FTTransformer(5, d_token=16, n_blocks=2, n_heads=4)
    out = m(torch.randn(8, 5))
    assert out.shape == (8,)


def test_pinn_loss_runs():
    m = PINNRegressor(5)
    x = torch.randn(16, 5)
    y = torch.randn(16)
    osnr = torch.rand(16)
    pred = m(x)
    total, dl, pl = pinn_loss(pred, y, osnr, lambda_phys=0.1)
    assert total.dim() == 0
    assert dl.item() >= 0 and pl.item() >= 0


def test_physics_anchor_monotone():
    osnr = torch.linspace(0.0, 1.0, 10)
    a = physics_anchor_q(osnr)
    diffs = a[1:] - a[:-1]
    assert (diffs > 0).all()


def test_regression_metrics_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.05, 3.95])
    m = regression_metrics(y, p)
    assert {"RMSE", "MAE", "R2", "MAPE"}.issubset(m.keys())


def test_split_conformal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 1000)
    p = y + rng.normal(0, 0.5, 1000)
    yt = rng.normal(0, 1, 500)
    pt = yt + rng.normal(0, 0.5, 500)
    lo, hi, q = split_conformal_intervals(y, p, pt, alpha=0.1)
    cov = coverage(yt, lo, hi)
    assert cov >= 0.85  # ~90% target with margin
