import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_ames_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    st.success(" Model dan artifacts berhasil dimuat!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🏠 Ames Housing Price Prediction")
st.markdown("Prediksi harga jual rumah berdasarkan fitur properti")

# Sidebar input
st.sidebar.header("Input Fitur")

# Fungsi untuk membuat input berdasarkan tipe data (sederhana: semua numerik)
# Karena semua fitur sudah numerik setelah encoding, kita buat number_input untuk setiap fitur
# Tapi terlalu banyak (sekitar 70+ fitur). Kita hanya tampilkan fitur penting.
# Gunakan feature importance dari model untuk memilih top 10 fitur.

# Hitung feature importance
importance = model.feature_importances_
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)
top_features = imp_df.head(15)['feature'].tolist()

st.sidebar.markdown("### Fitur Paling Penting")
user_input = {}
for feat in top_features:
    # Nilai default diambil dari median dataset (jika kita punya). Untuk demo, default 0.
    user_input[feat] = st.sidebar.number_input(feat, value=0.0, step=1.0, format="%.0f")

# Fitur lainnya diisi dengan 0 (atau median). Untuk demo sederhana.
for feat in feature_names:
    if feat not in user_input:
        user_input[feat] = 0.0

# Prediksi
if st.sidebar.button("Prediksi Harga"):
    # Buat dataframe dengan urutan fitur yang benar
    input_df = pd.DataFrame([user_input])[feature_names]
    # Scaling (jika model memerlukan scaling, tapi Random Forest tidak perlu)
    # input_scaled = scaler.transform(input_df) # tidak perlu untuk RF
    prediction = model.predict(input_df)[0]
    st.success(f"🏡 Prediksi Harga Rumah: **${prediction:,.0f}**")
    st.balloons()
else:
    st.info("Silakan masukkan nilai fitur di sidebar, lalu klik Prediksi.")