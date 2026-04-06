 <img src="./streamlit%20testing%20images/test1.jpg" alt="Test 1" width="400"/>

*Test 1: Streamlit testing*

<img src="./streamlit%20testing%20images/test2.jpg" alt="Test 2" width="400"/>

*Test 2: Streamlit testing*
 
 # Ames Housing Price Prediction

## Project Overview

This project is a complete **data science portfolio** that predicts house sale prices using the **Ames Housing dataset** (Ames, Iowa).  
It includes **exploratory data analysis (EDA)**, **feature engineering**, **machine learning modeling**, **hyperparameter tuning**, and an **interactive web app** built with Streamlit.

The best performing model is a **Random Forest regressor** that achieves an **R² of 0.9229** and a **Mean Absolute Error (MAE) of $15,307** on the test set.

## Dataset

- **Source**: Ames Housing Dataset (alternative to Boston Housing)
- **Number of rows**: 2,930
- **Number of initial features**: 82
- **Number of features after preprocessing**: 77
- **Target variable**: `SalePrice` (in US dollars)

The dataset contains information about:
- Overall quality and condition (`Overall Qual`, `Exter Qual`)
- Area sizes (basement, first floor, second floor, garage, lot)
- Year built and year remodeled
- Categorical features (neighborhood, house style, etc.)

## Data Preprocessing & Feature Engineering

- Dropped non‑informative columns (`Order`, `PID`)
- Removed columns with >50% missing values
- Imputed missing numerical values with **median**, categorical with **mode**
- Encoded categorical variables using **Label Encoding**
- Created new features:
  - `TotalSF` = Total Basement SF + 1st Flr SF + 2nd Flr SF
  - `HouseAge` = Yr Sold – Year Built
  - `IsRemodeled` = 1 if Year Remod/Add ≠ Year Built else 0
  - `TotalBath` = sum of full/half bathrooms (including basement)

## Modeling

Four regression models were trained and evaluated using **MAE**, **RMSE**, and **R²**:

| Model              | Test MAE | Test RMSE | Test R²  |
|--------------------|----------|-----------|----------|
| Linear Regression  | $20,439  | $33,578   | 0.8594   |
| Ridge              | $20,436  | $33,575   | 0.8594   |
| Lasso              | $20,444  | $33,581   | 0.8593   |
| Random Forest      | $15,392  | $24,902   | **0.9227** |


### Final Performance After Tuning

| Metric | Value |
|--------|-------|
| **Test MAE** | $15,307 |
| **Test R²** | 0.9229 |

The model explains **92.3%** of the variance in house prices.

### Feature Importance

Top 5 most important features (from the tuned Random Forest):

| Feature | Importance |
|---------|------------|
| Overall Qual | 0.4870 |
| TotalSF | 0.2960 |
| 2nd Flr SF | 0.0200 |
| Year Built | 0.0129 |
| HouseAge | 0.0128 |

---

### Business Insight

- **Overall quality** and **total square footage** are the dominant predictors of house price.  
- The model can estimate home values with an **average error of about $15,300**, making it suitable for real‑estate price estimation.



### Deployment
Deployment
An interactive web application was built using Streamlit.
It allows users to input property features and get a real‑time price prediction.

To run locally:

bash
streamlit run app.py

The app loads the saved best_ames_model.pkl, scaler.pkl, and feature_names.pkl.

## How to Reproduce


### 1. Clone and enter folder
git clone <repository-url>
cd <repository-folder>

### 2. Create virtual environment
python -m venv .venv

source .venv/bin/activate   # Linux/macOS

 .venv\Scripts\activate    # Windows

### 3. Run notebook
jupyter notebook project.ipynb

### 4. Run Streamlit app
streamlit run app.py
