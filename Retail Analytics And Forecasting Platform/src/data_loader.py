from __future__ import annotations

import pandas as pd

from .config import RAW_CSV, TRANSACTIONS_PARQUET


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_processed() -> pd.DataFrame:
    return pd.read_parquet(TRANSACTIONS_PARQUET)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["Product line"]).reset_index(drop=True)
    if out["Customer type"].isna().any():
        out["Customer type"] = out["Customer type"].fillna(out["Customer type"].mode().iloc[0])
    for col in ["Unit price", "Quantity"]:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    out["Total"] = (out["Unit price"] * out["Quantity"] * 1.05).round(4)
    out["cogs"] = (out["Unit price"] * out["Quantity"]).round(4)
    out["Tax 5%"] = (out["cogs"] * 0.05).round(4)
    out["gross income"] = out["Tax 5%"]
    out["Date"] = pd.to_datetime(out["Date"], format="%m/%d/%y", errors="coerce")
    out["Time"] = pd.to_datetime(out["Time"], format="%H:%M", errors="coerce").dt.time
    out = out.dropna(subset=["Date"]).reset_index(drop=True)
    return out
