from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.loaders import load_transactions, warn_if_missing
from src import config

st.title("What If Simulator")
st.caption(
    "Move the sliders and see projected monthly revenue against the historical baseline. "
    "Neutral settings reproduce the baseline exactly."
)

if warn_if_missing(config.TRANSACTIONS_PARQUET, "transactions.parquet"):
    st.stop()

df = load_transactions().copy()
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M").astype(str)

col1, col2 = st.columns(2)
with col1:
    price_mult = st.slider("Unit price multiplier", 0.7, 1.3, 1.0, 0.01)
    qty_mult = st.slider("Quantity multiplier", 0.7, 1.3, 1.0, 0.01)
with col2:
    member_share_target = st.slider("Member share target", 0.3, 0.8, float(df["IsMember"].mean()), 0.01)
    ewallet_share_target = st.slider(
        "Ewallet share target",
        0.1,
        0.6,
        float((df["Payment"] == "Ewallet").mean()),
        0.01,
    )

sim = df.copy()
sim["Unit price"] = sim["Unit price"] * price_mult
sim["Quantity"] = (sim["Quantity"] * qty_mult).clip(lower=1).round().astype(int)
sim["TaxFree"] = sim["Unit price"] * sim["Quantity"]
sim["Total"] = sim["TaxFree"] * 1.05

current_member = df["IsMember"].mean()
member_lift = (member_share_target - current_member) * 0.05
sim["Total"] = sim["Total"] * (1 + member_lift)

current_ew = (df["Payment"] == "Ewallet").mean()
ew_lift = (ewallet_share_target - current_ew) * 0.03
sim["Total"] = sim["Total"] * (1 + ew_lift)

baseline = df.groupby("Month")["Total"].sum().reset_index().rename(columns={"Total": "Baseline"})
projected = sim.groupby("Month")["Total"].sum().reset_index().rename(columns={"Total": "Projected"})
combo = baseline.merge(projected, on="Month")
combo["Delta"] = combo["Projected"] - combo["Baseline"]

c1, c2, c3 = st.columns(3)
c1.metric("Baseline revenue", f"${combo['Baseline'].sum():,.0f}")
c2.metric("Projected revenue", f"${combo['Projected'].sum():,.0f}")
c3.metric(
    "Delta",
    f"${combo['Delta'].sum():,.0f}",
    f"{(combo['Projected'].sum() / combo['Baseline'].sum() - 1) * 100:.1f}%",
)

fig = go.Figure()
fig.add_bar(x=combo["Month"], y=combo["Baseline"], name="Baseline", marker_color="#b08968")
fig.add_bar(x=combo["Month"], y=combo["Projected"], name="Projected", marker_color="#2f5d62")
fig.update_layout(barmode="group", yaxis_title="Revenue (USD)", plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Member and Ewallet sliders apply soft uplift multipliers calibrated from the observed "
    "ticket size advantage of members and the modest digital payment uplift. The price and "
    "quantity sliders apply directly to each transaction."
)
