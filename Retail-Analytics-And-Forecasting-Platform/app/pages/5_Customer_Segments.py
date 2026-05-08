from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.loaders import load_segment_profiles, load_segments, warn_if_missing
from app.components.theme import inject_sidebar_style
from src import config

inject_sidebar_style()

st.title("Customer Segments")

if warn_if_missing(config.SEGMENTS_PARQUET, "segments.parquet"):
    st.stop()
if warn_if_missing(config.SEGMENT_PROFILES_PARQUET, "segment_profiles.parquet"):
    st.stop()

seg = load_segments()
profile = load_segment_profiles().reset_index()

st.subheader("Segment profile")
st.dataframe(profile, use_container_width=True)

PALETTE = ["#2f5d62", "#7d9b76", "#c9a86a", "#b08968", "#5b7e91", "#3a6e5f"]
fig = px.scatter(
    seg,
    x="Recency",
    y="Monetary",
    color=seg["Segment"].astype(str),
    hover_data=["Frequency", "AvgRating", "Branch", "DominantProduct"],
    labels={"color": "Segment"},
    color_discrete_sequence=PALETTE,
)
fig.update_layout(plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Download segment assignments")
st.download_button(
    "Download CSV",
    seg.to_csv(index=False).encode("utf-8"),
    file_name="customer_segments.csv",
    mime="text/csv",
)
