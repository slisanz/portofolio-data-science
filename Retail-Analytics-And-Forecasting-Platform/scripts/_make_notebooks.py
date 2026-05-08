"""Generator script for the six analysis notebooks. Run once: python scripts/_make_notebooks.py"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def write_nb(name: str, cells: list) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NB_DIR / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")


BOOT = """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.viz import set_style, fmt_money

set_style()
RNG = np.random.default_rng(config.RANDOM_SEED)
"""


# ----- 01 EDA -----
n01 = [
    md("# 01 Exploratory Data Analysis\n\nMulti branch retail dataset, 1000 transactions across three supermarket branches in Myanmar. This notebook profiles the data, surfaces the strongest signals by branch, product line, payment method, customer type, and time, and persists a clean transactions table for downstream notebooks.\n"),
    code(BOOT),
    md("## Load the raw data\n"),
    code("""from src.data_loader import load_raw, basic_clean

raw = load_raw()
print('shape:', raw.shape)
raw.head()
"""),
    code("""raw.info()
"""),
    md("Missing values appear in `Product line` and `Customer type`. The cleaning step drops those rows and parses date and time columns.\n"),
    code("""raw.isna().sum().sort_values(ascending=False)
"""),
    code("""df = basic_clean(raw)
print('after clean:', df.shape)
df.head()
"""),
    md("## Descriptive statistics\n"),
    code("""df.describe(include='all').T.head(20)
"""),
    md("## Revenue by branch\n"),
    code("""rev_by_branch = df.groupby('Branch')['Total'].sum().sort_values(ascending=False)
print(rev_by_branch.apply(fmt_money))
ax = rev_by_branch.plot(kind='bar', color=['#1f6feb', '#10b981', '#f59e0b'])
ax.set_title('Total Revenue by Branch')
ax.set_ylabel('Revenue (USD)')
plt.tight_layout()
plt.show()
"""),
    md("## Revenue by product line\n"),
    code("""rev_by_pl = df.groupby('Product line')['Total'].sum().sort_values(ascending=True)
ax = rev_by_pl.plot(kind='barh', color='#1f6feb')
ax.set_title('Revenue by Product Line')
ax.set_xlabel('Revenue (USD)')
plt.tight_layout()
plt.show()
"""),
    md("## Customer type contribution\n"),
    code("""ct = df.groupby('Customer type').agg(Revenue=('Total', 'sum'), Transactions=('Invoice ID', 'count'), AvgTicket=('Total', 'mean'))
ct['Share'] = ct['Revenue'] / ct['Revenue'].sum()
ct
"""),
    md("## Payment mix\n"),
    code("""pay = df.groupby('Payment').agg(Revenue=('Total', 'sum'), Transactions=('Invoice ID', 'count'))
pay['Share'] = pay['Revenue'] / pay['Revenue'].sum()
ax = pay['Share'].plot(kind='bar', color='#10b981')
ax.set_title('Revenue Share by Payment Method')
ax.set_ylabel('Share')
plt.tight_layout()
plt.show()
pay
"""),
    md("## Monthly trend\n"),
    code("""monthly = df.assign(Month=df['Date'].dt.to_period('M').astype(str)).groupby('Month')['Total'].sum()
ax = monthly.plot(marker='o', color='#1f6feb')
ax.set_title('Monthly Revenue')
ax.set_ylabel('Revenue (USD)')
plt.tight_layout()
plt.show()
monthly.apply(fmt_money)
"""),
    md("## Rating distribution by branch\n"),
    code("""ax = sns.boxplot(data=df, x='Branch', y='Rating', palette=['#1f6feb', '#10b981', '#f59e0b'])
ax.set_title('Customer Rating by Branch')
plt.tight_layout()
plt.show()
"""),
    md("## Persist the cleaned table\nThe feature engineering notebook reads this artifact and adds derived columns.\n"),
    code("""config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
df.to_parquet(config.DATA_PROCESSED / 'transactions_clean.parquet', index=False)
print('written:', config.DATA_PROCESSED / 'transactions_clean.parquet')
"""),
    md("## Findings\n\n1. Branch C posts the highest revenue and gross income.\n2. Fashion accessories and Food and beverages dominate the revenue ranking, while Food and beverages also leads on customer rating.\n3. Members contribute the larger share of revenue and have a higher average basket than non members.\n4. Cash and Ewallet are nearly tied on revenue, signalling room to migrate volume into digital rails.\n5. January is the strongest month and a target for stocking and promotion planning.\n"),
]
write_nb("01_eda.ipynb", n01)


# ----- 02 Feature Engineering -----
n02 = [
    md("# 02 Feature Engineering\n\nDerive temporal, customer, and basket features from the cleaned transactions. Output is `data/processed/transactions.parquet`, the canonical table consumed by notebooks 03 to 06 and the Streamlit app.\n"),
    code(BOOT),
    code("""from src.data_loader import load_raw, basic_clean
from src.features import engineer

raw = load_raw()
clean = basic_clean(raw)
df = engineer(clean)
print('shape:', df.shape)
df.head()
"""),
    md("## Sanity checks on derived columns\n"),
    code("""df[['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsMember', 'IsFemale', 'BasketTier']].describe(include='all').T
"""),
    md("## Hour of day pattern\n"),
    code("""hourly = df.groupby('Hour')['Total'].sum()
ax = hourly.plot(marker='o', color='#1f6feb')
ax.set_title('Revenue by Hour of Day')
ax.set_ylabel('Revenue (USD)')
plt.tight_layout(); plt.show()
"""),
    md("## Weekend vs weekday\n"),
    code("""wk = df.groupby('IsWeekend').agg(Revenue=('Total', 'sum'), Transactions=('Invoice ID', 'count'), AvgTicket=('Total', 'mean'))
wk.index = ['Weekday', 'Weekend']
wk
"""),
    md("## Persist the engineered table\n"),
    code("""config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
df.to_parquet(config.TRANSACTIONS_PARQUET, index=False)
print('written:', config.TRANSACTIONS_PARQUET, 'rows:', len(df))
"""),
]
write_nb("02_feature_engineering.ipynb", n02)


# ----- 03 Customer Segmentation -----
n03 = [
    md("# 03 Customer Segmentation\n\nApproximate RFM features at the invoice level (the dataset has no repeat customer identifier, so each invoice is treated as a customer touchpoint), then run KMeans across a small grid of cluster counts and select the best by silhouette score. Persist the model, scaler, and labelled table.\n"),
    code(BOOT),
    code("""from src.data_loader import load_processed
from src.features import rfm_table
from src.modelling import fit_kmeans
import joblib

df = load_processed()
rfm = rfm_table(df)
print('rfm shape:', rfm.shape)
rfm.head()
"""),
    md("## Fit KMeans across k\n"),
    code("""result = fit_kmeans(rfm, k_range=range(3, 7))
print('best k:', result.k, 'silhouette:', round(result.silhouette, 3))
rfm['Segment'] = result.labels
"""),
    md("## Profile each segment\n"),
    code("""profile = rfm.groupby('Segment').agg(
    Customers=('Invoice ID', 'count'),
    AvgRecency=('Recency', 'mean'),
    AvgFrequency=('Frequency', 'mean'),
    AvgMonetary=('Monetary', 'mean'),
    AvgRating=('AvgRating', 'mean'),
).round(2)
profile['Share'] = profile['Customers'] / profile['Customers'].sum()
profile
"""),
    md("## Visualise the segments\n"),
    code("""ax = sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='tab10', alpha=0.7)
ax.set_title('Segments in Recency vs Monetary space')
plt.tight_layout(); plt.show()
"""),
    md("## Save model artifacts\n"),
    code("""config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(result.model, config.KMEANS_MODEL)
joblib.dump(result.scaler, config.SCALER_MODEL)
rfm.to_parquet(config.SEGMENTS_PARQUET, index=False)
profile.to_parquet(config.SEGMENT_PROFILES_PARQUET)
print('saved:', config.KMEANS_MODEL, config.SEGMENTS_PARQUET, config.SEGMENT_PROFILES_PARQUET)
"""),
    md("## Action recommendations\n\n1. The highest monetary, lowest recency segment is the loyalty target; route them into a paid membership tier.\n2. The lowest monetary, highest recency segment is the churn risk; reactivate with targeted promotions on Food and beverages where the rating is highest.\n3. Mid tiers are the volume cohort and respond best to bundling and weekday promotions.\n"),
]
write_nb("03_customer_segmentation.ipynb", n03)


# ----- 04 Sales Forecasting -----
n04 = [
    md("# 04 Sales Forecasting\n\nForecast daily revenue per branch with Prophet (primary) and ARIMA (baseline). Use a small walk forward cross validation to score MAE and MAPE, pick the winner per branch, and emit a 30 day forecast with confidence band.\n"),
    code(BOOT),
    code("""from src.data_loader import load_processed
from src.modelling import regression_metrics, walk_forward_split
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import joblib
import warnings
warnings.filterwarnings('ignore')

df = load_processed()
df['Date'] = pd.to_datetime(df['Date'])
"""),
    md("## Build daily series per branch\n"),
    code("""daily = df.groupby(['Branch', 'Date'])['Total'].sum().reset_index()
daily = daily.rename(columns={'Date': 'ds', 'Total': 'y'})
print(daily.groupby('Branch')['y'].agg(['count', 'mean']))
"""),
    md("## Fit and score per branch\n"),
    code("""def fit_prophet(frame):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    m.fit(frame)
    return m

def score_prophet(train, test):
    frame = train.reset_index()
    frame.columns = ['ds', 'y']
    m = fit_prophet(frame)
    future = pd.DataFrame({'ds': test.index})
    pred = m.predict(future)['yhat'].values
    return regression_metrics(test.values, pred), pred

def score_arima(train, test):
    model = ARIMA(train.values, order=(2, 1, 2)).fit()
    pred = model.forecast(steps=len(test))
    return regression_metrics(test.values, pred), pred

forecasts = []
metrics_rows = []
models_per_branch = {}

for branch in config.BRANCHES:
    series = daily[daily['Branch'] == branch].set_index('ds')['y'].asfreq('D').fillna(0)
    splits = walk_forward_split(series, n_splits=3, horizon=7)
    p_mae, a_mae = [], []
    for tr, te in splits:
        try:
            mp, _ = score_prophet(tr, te); p_mae.append(mp['mae'])
        except Exception as e:
            print(f'prophet fold failed for {branch}:', e)
        try:
            ma, _ = score_arima(tr, te); a_mae.append(ma['mae'])
        except Exception as e:
            print(f'arima fold failed for {branch}:', e)
    p_avg = float(np.mean(p_mae)) if p_mae else np.inf
    a_avg = float(np.mean(a_mae)) if a_mae else np.inf
    winner = 'prophet' if p_avg <= a_avg else 'arima'
    metrics_rows.append({'branch': branch, 'prophet_mae': p_avg, 'arima_mae': a_avg, 'winner': winner})

    full = series.reset_index()
    full.columns = ['ds', 'y']
    try:
        m = fit_prophet(full)
        future = m.make_future_dataframe(periods=30)
        fc = m.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        models_per_branch[branch] = m
    except Exception as e:
        print(f'prophet final fit failed for {branch}, falling back to ARIMA:', e)
        ar = ARIMA(series.values, order=(2, 1, 2)).fit()
        steps = 30
        idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=steps, freq='D')
        full_idx = list(series.index) + list(idx)
        in_sample = ar.predict(start=0, end=len(series) - 1)
        fcst = ar.forecast(steps=steps)
        yhat = list(in_sample) + list(fcst)
        fc = pd.DataFrame({'ds': full_idx, 'yhat': yhat})
        fc['yhat_lower'] = fc['yhat'] * 0.85
        fc['yhat_upper'] = fc['yhat'] * 1.15
        winner = 'arima'
    fc['branch'] = branch
    fc['model'] = winner
    forecasts.append(fc)

metrics = pd.DataFrame(metrics_rows)
metrics
"""),
    md("## Plot forecasts\n"),
    code("""fc_all = pd.concat(forecasts, ignore_index=True)
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, branch in zip(axes, config.BRANCHES):
    sub = fc_all[fc_all['branch'] == branch]
    ax.plot(sub['ds'], sub['yhat'], color='#1f6feb', label='forecast')
    ax.fill_between(sub['ds'], sub['yhat_lower'], sub['yhat_upper'], color='#1f6feb', alpha=0.15)
    obs = daily[daily['Branch'] == branch]
    ax.scatter(obs['ds'], obs['y'], s=10, color='#0f172a', alpha=0.6, label='actual')
    ax.set_title(f'Branch {branch}')
    ax.legend(loc='upper left')
plt.tight_layout(); plt.show()
"""),
    md("## Persist forecasts and models\n"),
    code("""fc_all.to_parquet(config.FORECAST_PARQUET, index=False)
metrics.to_parquet(config.DATA_PROCESSED / 'forecast_metrics.parquet', index=False)
for branch, m in models_per_branch.items():
    try:
        joblib.dump(m, config.PROPHET_MODEL(branch))
    except Exception as e:
        print(f'could not pickle prophet model for {branch}:', e)
print('saved forecast and prophet models')
"""),
]
write_nb("04_sales_forecasting.ipynb", n04)


# ----- 05 Rating Prediction -----
n05 = [
    md("# 05 Rating Prediction\n\nPredict customer rating from transaction context. Compare a linear baseline against gradient boosting, report MAE and RMSE on a held out split, and explain predictions globally and locally with SHAP.\n"),
    code(BOOT),
    code("""from src.data_loader import load_processed
from src.modelling import regression_metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

df = load_processed()
"""),
    md("## Feature matrix\n"),
    code("""num = ['Unit price', 'Quantity', 'Total', 'Hour', 'IsWeekend', 'IsMember', 'IsFemale']
cat = ['Branch', 'Product line', 'Payment']
X = df[num + cat]
y = df['Rating']

pre = ColumnTransformer([
    ('num', StandardScaler(), num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
])
"""),
    md("## Train and score\n"),
    code("""X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED)

ridge = Pipeline([('pre', pre), ('m', Ridge(alpha=1.0, random_state=config.RANDOM_SEED))]).fit(X_tr, y_tr)
gbr = Pipeline([('pre', pre), ('m', GradientBoostingRegressor(random_state=config.RANDOM_SEED))]).fit(X_tr, y_tr)

ridge_metrics = regression_metrics(y_te, ridge.predict(X_te))
gbr_metrics = regression_metrics(y_te, gbr.predict(X_te))
metrics_tbl = pd.DataFrame({'ridge': ridge_metrics, 'gbr': gbr_metrics}).T
metrics_tbl
"""),
    md("## SHAP explanations\nSHAP picks TreeExplainer for the gradient boosting model and LinearExplainer for the ridge baseline. We pass a transformed background sample so SHAP can model feature dependencies.\n"),
    code("""import shap

best = gbr if gbr_metrics['mae'] <= ridge_metrics['mae'] else ridge
X_te_t = best.named_steps['pre'].transform(X_te)
X_te_arr = X_te_t.toarray() if hasattr(X_te_t, 'toarray') else X_te_t
feature_names = list(best.named_steps['pre'].get_feature_names_out())

try:
    explainer = shap.Explainer(best.named_steps['m'], X_te_arr)
except Exception:
    explainer = shap.Explainer(best.named_steps['m'].predict, X_te_arr)

shap_values = explainer(X_te_arr)
if hasattr(shap_values, 'feature_names'):
    shap_values.feature_names = feature_names
shap.plots.beeswarm(shap_values, max_display=12, show=True)
"""),
    md("## Persist artifacts\n"),
    code("""config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best, config.RATING_MODEL)
metrics_tbl.to_parquet(config.RATING_METRICS_PARQUET)

shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_df.to_parquet(config.RATING_SHAP_PARQUET, index=False)
print('saved rating model, metrics, shap values')
"""),
]
write_nb("05_rating_prediction.ipynb", n05)


# ----- 06 Market Basket -----
n06 = [
    md("# 06 Market Basket Analysis\n\nThe dataset records one product line per invoice, so a classical multi item Apriori run requires building artificial baskets. We aggregate by Date and Branch to capture co occurrence of product lines within the same day at the same branch, then run Apriori and surface the strongest lift rules for bundling decisions.\n"),
    code(BOOT),
    code("""from src.data_loader import load_processed
from mlxtend.frequent_patterns import apriori, association_rules

df = load_processed()
"""),
    md("## Build daily branch baskets\n"),
    code("""baskets = (
    df.assign(flag=1)
      .pivot_table(index=['Branch', 'Date'], columns='Product line', values='flag', aggfunc='max', fill_value=0)
)
baskets.head()
"""),
    md("## Run Apriori\nWe scan a small grid of support thresholds and keep the lowest threshold that yields at least one rule, so the notebook is robust on the small sample.\n"),
    code("""freq = pd.DataFrame()
rules = pd.DataFrame()
for support in [0.20, 0.15, 0.10, 0.05, 0.03]:
    freq = apriori(baskets.astype(bool), min_support=support, use_colnames=True)
    if freq.empty:
        continue
    candidate = association_rules(freq, metric='lift', min_threshold=1.0)
    if not candidate.empty:
        rules = candidate.sort_values('lift', ascending=False)
        print('using support =', support, 'rules =', len(rules))
        break
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15)
"""),
    md("## Persist rules\n"),
    code("""out = rules.copy() if not rules.empty else pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
if not out.empty:
    out['antecedents'] = out['antecedents'].apply(lambda s: ', '.join(sorted(s)))
    out['consequents'] = out['consequents'].apply(lambda s: ', '.join(sorted(s)))
out[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_parquet(config.RULES_PARQUET, index=False)
print('saved:', config.RULES_PARQUET)
"""),
]
write_nb("06_market_basket.ipynb", n06)


print('all notebooks generated under', NB_DIR)
