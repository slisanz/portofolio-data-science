import numpy as np

from src.data import load_raw, make_splits, split_xy
from src.config import FEATURE_COLS, TARGET_COL


def test_load_raw_schema():
    df = load_raw(nrows=1000)
    assert len(df) == 1000
    assert list(df.columns) == FEATURE_COLS + [TARGET_COL]
    assert df.isna().sum().sum() == 0


def test_split_proportions():
    df = load_raw(nrows=10_000)
    tr, va, te = make_splits(df)
    total = len(tr) + len(va) + len(te)
    assert total == len(df)
    assert abs(len(tr) / total - 0.70) < 0.02
    assert abs(len(va) / total - 0.15) < 0.02
    assert abs(len(te) / total - 0.15) < 0.02


def test_split_xy_shapes():
    df = load_raw(nrows=500)
    X, y = split_xy(df)
    assert X.shape == (500, len(FEATURE_COLS))
    assert y.shape == (500,)
