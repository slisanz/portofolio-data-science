from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    return pd.read_parquet(config.TRANSACTIONS_PARQUET)


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame:
    return pd.read_parquet(config.SEGMENTS_PARQUET)


@st.cache_data(show_spinner=False)
def load_segment_profiles() -> pd.DataFrame:
    return pd.read_parquet(config.SEGMENT_PROFILES_PARQUET)


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    return pd.read_parquet(config.FORECAST_PARQUET)


@st.cache_data(show_spinner=False)
def load_forecast_metrics() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_PROCESSED / "forecast_metrics.parquet")


@st.cache_data(show_spinner=False)
def load_rules() -> pd.DataFrame:
    return pd.read_parquet(config.RULES_PARQUET)


@st.cache_data(show_spinner=False)
def load_rating_metrics() -> pd.DataFrame:
    return pd.read_parquet(config.RATING_METRICS_PARQUET)


@st.cache_resource(show_spinner=False)
def load_rating_model():
    return joblib.load(config.RATING_MODEL)


def warn_if_missing(path: Path, label: str) -> bool:
    if not path.exists():
        st.warning(
            f"Missing artifact: {label}. Run the notebooks first, see README for the order."
        )
        return True
    return False
