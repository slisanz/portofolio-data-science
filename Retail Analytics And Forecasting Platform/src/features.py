from __future__ import annotations

import numpy as np
import pandas as pd


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_str = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    time_str = out["Time"].astype(str).str.slice(0, 8)
    out["DateTime"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
    out["Hour"] = out["DateTime"].dt.hour
    out["DayOfWeek"] = out["DateTime"].dt.day_name()
    out["DayOfWeekIdx"] = out["DateTime"].dt.dayofweek
    out["Month"] = out["DateTime"].dt.month_name()
    out["MonthIdx"] = out["DateTime"].dt.month
    out["IsWeekend"] = out["DayOfWeekIdx"].isin([5, 6]).astype(int)
    out["IsMember"] = (out["Customer type"] == "Member").astype(int)
    out["IsFemale"] = (out["Gender"] == "Female").astype(int)
    out["UnitMargin"] = out["Total"] / out["Quantity"].replace(0, np.nan)
    out["BasketTier"] = pd.cut(
        out["Total"],
        bins=[-1, 100, 300, 500, 1e9],
        labels=["Low", "Mid", "High", "VeryHigh"],
    ).astype(str)
    return out


def rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Invoice level RFM proxy.

    The dataset has no repeat customer identifier, so each invoice is treated as a
    distinct customer touchpoint. Frequency is replaced with Quantity (items in the
    basket) which varies meaningfully across invoices and serves as a useful loyalty
    proxy alongside Recency and Monetary.
    """
    snapshot = df["DateTime"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("Invoice ID")
        .agg(
            Recency=("DateTime", lambda s: (snapshot - s.max()).days),
            Frequency=("Quantity", "sum"),
            Monetary=("Total", "sum"),
            AvgRating=("Rating", "mean"),
            Branch=("Branch", "first"),
            CustomerType=("Customer type", "first"),
            DominantProduct=("Product line", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    return rfm
