import numpy as np

from src.rec_utils import mae, rmse


def test_rmse_zero():
    y = np.array([3.0, 4.0, 5.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_rmse_known():
    y = np.array([1.0, 2.0])
    p = np.array([2.0, 4.0])
    assert abs(rmse(y, p) - np.sqrt((1 + 4) / 2)) < 1e-9
    assert abs(mae(y, p) - 1.5) < 1e-9
