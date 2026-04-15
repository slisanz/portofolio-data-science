from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def test_movies_parquet_shape():
    df = pl.read_parquet(PROC / "movies.parquet")
    assert df.height > 80_000
    assert {"movieId", "title", "genres"}.issubset(df.columns)


def test_splits_exist():
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        p = PROC / "splits" / name
        assert p.exists(), f"missing {p}"
