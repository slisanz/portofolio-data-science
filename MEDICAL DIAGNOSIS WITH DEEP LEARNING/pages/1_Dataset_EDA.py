"""Dataset and EDA page.

Reads pre-computed artefacts from `reports/` so the page renders quickly even when the
dataset folder is not present (for example after deploying to Streamlit Community Cloud
without uploading the X-rays).
"""

from pathlib import Path

import json
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

st.set_page_config(page_title="Dataset and EDA", layout="wide")
st.title("Dataset and Exploratory Analysis")
st.caption(
    "Pediatric chest X-rays (anterior-posterior view) from Guangzhou Women & Children's "
    "Medical Center. Source: Kermany et al., 2018."
)


@st.cache_data
def load_summary():
    p = REPORTS_DIR / "eda_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_manifest():
    p = REPORTS_DIR / "image_manifest.csv"
    return pd.read_csv(p) if p.exists() else None


summary = load_summary()
manifest = load_manifest()

st.subheader("Headline numbers")
if summary is None:
    st.info("Run `notebooks/01_eda.ipynb` to generate `reports/eda_summary.json`.")
else:
    a, b, c = st.columns(3)
    a.metric("Total images", f"{summary['total_images']:,}")
    b.metric("Train imbalance ratio", f"{summary['train_imbalance_ratio']}x")
    c.metric("Classes", "NORMAL, PNEUMONIA")
    st.json(summary, expanded=False)

st.subheader("Class distribution per split")
fig_path = FIG_DIR / "class_distribution.png"
if fig_path.exists():
    st.image(str(fig_path), use_column_width=True)
else:
    st.info("Figure not yet generated. Run the EDA notebook.")

st.subheader("Pneumonia subtype distribution")
sub_path = FIG_DIR / "subtype_distribution.png"
if sub_path.exists():
    st.image(str(sub_path), use_column_width=True)

st.subheader("Image dimensions")
dim_path = FIG_DIR / "image_dimensions.png"
if dim_path.exists():
    st.image(str(dim_path), use_column_width=True)

st.subheader("Mean pixel intensity by class")
int_path = FIG_DIR / "intensity_distribution.png"
if int_path.exists():
    st.image(str(int_path), use_column_width=True)

st.subheader("Sample images")
sample_cols = st.columns(3)
for col, name, label in [
    (sample_cols[0], "samples_normal.png", "Normal"),
    (sample_cols[1], "samples_bacteria.png", "Bacterial pneumonia"),
    (sample_cols[2], "samples_virus.png", "Viral pneumonia"),
]:
    p = FIG_DIR / name
    if p.exists():
        col.image(str(p), caption=label, use_column_width=True)

if manifest is not None:
    with st.expander("Browse manifest dataframe"):
        st.dataframe(manifest, use_container_width=True, hide_index=True)
