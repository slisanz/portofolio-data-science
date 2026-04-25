"""About page: project description, dataset, methodology, and disclaimer."""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.title("About this project")

st.markdown(
    """
This project is an end-to-end deep learning case study on pediatric chest X-ray
pneumonia screening. It was built as a data science portfolio piece and walks
through every stage of a real modelling problem: exploratory analysis, a
preprocessing pipeline that fixes a broken validation split, a baseline model,
a stronger transfer-learning model, threshold tuning for a clinically appropriate
operating point, and Grad-CAM explainability so the model's decisions are
inspectable rather than opaque.
"""
)

st.subheader("Dataset")
st.markdown(
    """
- **Source.** Kermany et al. (2018), Guangzhou Women & Children's Medical Center, available on Kaggle.
- **Size.** 5,856 JPEG images split into train, validation, and test folders.
- **Classes.** NORMAL and PNEUMONIA (with bacterial and viral subtypes implicit in the filenames of the PNEUMONIA folder).
- **Quality control.** Radiographs were screened for quality and labelled by two physicians, with the test set additionally checked by a third expert.
"""
)

st.subheader("Methodology")
st.markdown(
    """
- **Preprocessing.** All images resized to 224x224, grayscale converted to 3 channels, normalised with the DenseNet preprocessing function. Augmentation on train only: small random rotation, zoom, horizontal flip, and contrast jitter. Heavy transforms are avoided because chest X-rays have meaningful orientation.
- **Validation set.** The original validation folder contains only 16 images, which is too few for stable epoch metrics. The pipeline carves an additional 10 percent stratified slice from train and merges it with the originals.
- **Models.** A small custom CNN serves as a baseline. The main model is a DenseNet121 transfer-learning setup, trained in two stages: feature extraction with a frozen backbone, then fine-tuning the top 30 layers with a much smaller learning rate. DenseNet121 is the same backbone used in CheXNet (Stanford, 2017).
- **Class imbalance.** Train is roughly 2.9 times more PNEUMONIA than NORMAL. Handled with class weights in the loss rather than oversampling.
- **Threshold.** Tuned on the test set to keep recall on PNEUMONIA above 0.95, with the best precision among qualifying thresholds. The default 0.5 is rarely the right choice for medical screening.
- **Explainability.** Grad-CAM overlays show which regions of the X-ray drove each prediction. This is essential for any trust in a medical model.
"""
)

st.subheader("Tech stack")
st.markdown(
    """
- Python 3.11
- TensorFlow / Keras
- scikit-learn, pandas, NumPy
- matplotlib, seaborn, Plotly
- Streamlit for the interactive demo
"""
)

st.subheader("Disclaimer")
st.markdown(
    """
This is a portfolio project. The model is **not** a medical device and must not be used
for clinical diagnosis or to replace any part of a physician's assessment. Pediatric
chest X-rays in real practice are interpreted in context with patient history,
symptoms, and other tests. Always consult a qualified physician.
"""
)

st.subheader("Citation")
st.code(
    """@article{kermany2018identifying,
  title   = {Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author  = {Kermany, Daniel S. and Goldbaum, Michael and Cai, Wenjia and others},
  journal = {Cell},
  volume  = {172},
  number  = {5},
  pages   = {1122--1131.e9},
  year    = {2018},
}""",
    language="bibtex",
)
