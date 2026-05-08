from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.loaders import load_transactions, warn_if_missing
from app.components.theme import inject_sidebar_style
from src import config

inject_sidebar_style()

st.title("Branch Compare")

if warn_if_missing(config.TRANSACTIONS_PARQUET, "transactions.parquet"):
    st.stop()

df = load_transactions()

agg = df.groupby("Branch").agg(
    Revenue=("Total", "sum"),
    Transactions=("Invoice ID", "count"),
    AvgTicket=("Total", "mean"),
    AvgRating=("Rating", "mean"),
    GrossIncome=("gross income", "sum"),
).round(2)
agg["City"] = agg.index.map(config.CITY_BY_BRANCH)
st.dataframe(agg, use_container_width=True)

PALETTE = ["#2f5d62", "#7d9b76", "#c9a86a", "#b08968"]
fig = px.bar(agg.reset_index(), x="Branch", y="Revenue", color="City", text_auto=".2s", color_discrete_sequence=PALETTE)
fig.update_layout(plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Revenue difference test")
groups = [df.loc[df["Branch"] == b, "Total"].values for b in config.BRANCHES]
f_stat, p_val = stats.f_oneway(*groups)
st.write(
    f"One way ANOVA across branches: F = {f_stat:.3f}, p value = {p_val:.4f}."
)
if p_val < 0.05:
    st.info("At alpha 0.05 the branch revenue means differ significantly.")
else:
    st.info("At alpha 0.05 there is no significant difference between branch revenue means.")

st.subheader("Rating distribution")
fig2 = px.box(df, x="Branch", y="Rating", color="Branch", color_discrete_sequence=PALETTE)
fig2.update_layout(plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec")
st.plotly_chart(fig2, use_container_width=True)
