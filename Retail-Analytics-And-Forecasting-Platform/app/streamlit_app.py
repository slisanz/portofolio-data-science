from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Retail Analytics And Forecasting Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Retail Analytics And Forecasting Platform")
st.caption(
    "Data engineering (pandas) plus RFM segmentation (KMeans) plus dual forecasting "
    "(Prophet vs ARIMA, walk forward CV) plus rating regression (Ridge plus Gradient Boosting) "
    "plus SHAP explainability plus market basket (Apriori) plus Streamlit dashboard with "
    "interactive what if simulator."
)

st.markdown(
    """
This dashboard summarises a portfolio data science study on a 1000 transaction
supermarket dataset spanning three branches in Myanmar. Use the sidebar to navigate
between pages.

Pages
1. Rating Predictor, predict customer rating and inspect SHAP contributions.
2. What If Simulator, move sliders and see projected revenue against baseline.
3. Overview, headline KPIs and revenue mix.
4. Branch Compare, side by side branch level metrics with a statistical test.
5. Customer Segments, KMeans cohorts with profile cards and a downloadable export.
6. Sales Forecast, Prophet forecasts with cross validation metrics.

All artifacts are produced by the notebooks under `notebooks/`. The app loads
parquet and joblib outputs and does no training at runtime.
"""
)

st.sidebar.success("Pick a page above.")
