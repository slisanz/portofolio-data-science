from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PALETTE = ["#2f5d62", "#7d9b76", "#c9a86a", "#b08968", "#5b7e91", "#3a6e5f"]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.loaders import load_transactions, warn_if_missing
from src import config

st.title("Overview")

if warn_if_missing(config.TRANSACTIONS_PARQUET, "transactions.parquet"):
    st.stop()

df = load_transactions()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total revenue", f"${df['Total'].sum():,.0f}")
c2.metric("Transactions", f"{len(df):,}")
c3.metric("Average rating", f"{df['Rating'].mean():.2f}")
c4.metric("Gross income", f"${df['gross income'].sum():,.0f}")

st.subheader("Revenue trend")
df["Date"] = pd.to_datetime(df["Date"])
trend = df.groupby("Date")["Total"].sum().reset_index()
fig = px.line(trend, x="Date", y="Total", labels={"Total": "Revenue (USD)"}, color_discrete_sequence=[PALETTE[0]])
fig.update_layout(plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Revenue by product line")
    pl = df.groupby("Product line")["Total"].sum().reset_index().sort_values("Total")
    fig2 = px.bar(pl, x="Total", y="Product line", orientation="h", color_discrete_sequence=[PALETTE[0]])
    fig2.update_layout(plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec")
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    st.subheader("Payment mix")
    pay = df.groupby("Payment")["Total"].sum().reset_index()
    fig3 = px.pie(pay, names="Payment", values="Total", hole=0.4, color_discrete_sequence=PALETTE)
    fig3.update_layout(paper_bgcolor="#f7f3ec")
    st.plotly_chart(fig3, use_container_width=True)
