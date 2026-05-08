from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.loaders import load_rating_metrics, load_rating_model, warn_if_missing
from src import config

st.title("Rating Predictor")

if warn_if_missing(config.RATING_MODEL, "rating_model.joblib"):
    st.stop()

model = load_rating_model()
metrics = load_rating_metrics()

st.subheader("Model performance on hold out")
st.dataframe(metrics, use_container_width=True)

st.subheader("Predict a rating")
col1, col2, col3 = st.columns(3)
with col1:
    branch = st.selectbox("Branch", config.BRANCHES)
    payment = st.selectbox("Payment", config.PAYMENT_METHODS)
    product = st.selectbox("Product line", config.PRODUCT_LINES)
with col2:
    unit_price = st.number_input("Unit price", 5.0, 200.0, 50.0, step=1.0)
    quantity = st.number_input("Quantity", 1, 20, 5)
    hour = st.slider("Hour of day", 9, 21, 14)
with col3:
    is_member = st.checkbox("Member", value=True)
    is_female = st.checkbox("Female customer", value=True)
    is_weekend = st.checkbox("Weekend", value=False)

total = unit_price * quantity * 1.05
row = pd.DataFrame(
    [
        {
            "Unit price": unit_price,
            "Quantity": quantity,
            "Total": total,
            "Hour": hour,
            "IsWeekend": int(is_weekend),
            "IsMember": int(is_member),
            "IsFemale": int(is_female),
            "Branch": branch,
            "Product line": product,
            "Payment": payment,
        }
    ]
)
pred = float(model.predict(row)[0])
st.metric("Predicted rating (0 to 10)", f"{pred:.2f}")
st.caption(f"Implied basket value: ${total:,.2f}")
