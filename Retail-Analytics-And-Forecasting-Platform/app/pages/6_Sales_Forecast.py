from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.loaders import (
    load_forecast,
    load_forecast_metrics,
    load_transactions,
    warn_if_missing,
)
from app.components.theme import inject_sidebar_style
from src import config

inject_sidebar_style()

st.title("Sales Forecast")

if warn_if_missing(config.FORECAST_PARQUET, "forecast.parquet"):
    st.stop()

fc = load_forecast()
metrics = load_forecast_metrics()
df = load_transactions()
df["Date"] = pd.to_datetime(df["Date"])

branch = st.selectbox("Branch", config.BRANCHES, index=2)
horizon = st.slider("Forecast horizon (days)", 7, 30, 30, step=1)

sub = fc[fc["branch"] == branch].copy()
sub["ds"] = pd.to_datetime(sub["ds"])
last_actual = df["Date"].max()
sub = sub[sub["ds"] <= last_actual + pd.Timedelta(days=horizon)]

obs = df[df["Branch"] == branch].groupby("Date")["Total"].sum().reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=obs["Date"], y=obs["Total"], mode="markers", name="actual", marker=dict(color="#1f2937", size=6)))
fig.add_trace(go.Scatter(x=sub["ds"], y=sub["yhat"], mode="lines", name="forecast", line=dict(color="#2f5d62", width=2)))
fig.add_trace(go.Scatter(x=sub["ds"], y=sub["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=sub["ds"], y=sub["yhat_lower"], mode="lines", fill="tonexty", fillcolor="rgba(125,155,118,0.25)", line=dict(width=0), name="confidence"))
fig.update_layout(xaxis_title="Date", yaxis_title="Revenue (USD)", plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Cross validation metrics")
st.dataframe(metrics, use_container_width=True)
