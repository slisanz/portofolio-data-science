import nbformat

# ===== nb02 feature engineering =====
nb = nbformat.read('notebooks/02_feature_engineering.ipynb', as_version=4)
nb.cells[0].source = """# 02 Feature Engineering

The raw schema is transaction wide but model thin. Three of the four downstream notebooks (segmentation, forecasting, rating) need temporal, customer, and basket features that do not exist in the source CSV. This notebook builds them once, persists the result to `data/processed/transactions.parquet`, and lets every later notebook read a single tidy artifact.

The features fall into four groups:

| Group | New columns | Why we need them |
|-------|-------------|------------------|
| Temporal | `DateTime`, `Hour`, `DayOfWeek`, `DayOfWeekIdx`, `Month`, `MonthIdx`, `IsWeekend` | Forecasting needs a daily index. Rating prediction needs hour and weekend signals. The dashboard needs month labels for the trend chart. |
| Customer | `IsMember`, `IsFemale` | Boolean encodings for downstream linear and tree models, easier to read than one hot dummies for a binary. |
| Basket | `UnitMargin`, `BasketTier` | Per item value (revenue per quantity) and a coarse high to low ticket bucket used by the segment recommendations. |
| Identity | (already present) `Branch`, `City`, `Product line`, `Payment` | Carried through unchanged so the engineered table is a superset of the cleaned table. |

The cleaned input has zero NaN (verified at the end of notebook 01), so feature engineering is purely additive. No row is dropped here."""

explain_md = nbformat.v4.new_markdown_cell("The describe table below is a sanity check, not analysis. Each derived column should fall in the range its definition implies: `Hour` between 10 and 20 (mall opening hours), `IsWeekend` and member or gender flags between 0 and 1, and `BasketTier` taking one of four ordinal labels.")
nb.cells.insert(4, explain_md)

persist_explain = nbformat.v4.new_markdown_cell("The output parquet has 28 columns: the original 17 from `transactions_clean.parquet` plus 11 engineered features. Notebooks 03 to 06 and the Streamlit app read this single file. If the schema needs to change, this is the only notebook to touch.")
for i, c in enumerate(nb.cells):
    if c.cell_type == 'markdown' and c.source.startswith('## Persist'):
        nb.cells.insert(i + 1, persist_explain)
        break

nbformat.write(nb, 'notebooks/02_feature_engineering.ipynb')
print('nb02 patched')

# ===== nb03 segmentation =====
nb = nbformat.read('notebooks/03_customer_segmentation.ipynb', as_version=4)
nb.cells[0].source = """# 03 Customer Segmentation

Group invoices into behavioural segments so the business can target campaigns instead of blasting every customer with the same offer. The classical tool here is RFM (Recency, Frequency, Monetary) followed by KMeans.

This dataset does not carry a repeat customer identifier, so we cannot compute true Frequency the textbook way. The workaround:

1. Treat every invoice as one customer touchpoint.
2. **Recency** = days from the invoice date to one day past the dataset max.
3. **Frequency** is replaced with **Quantity** (items in the basket). It still varies meaningfully and acts as a loyalty proxy.
4. **Monetary** = invoice total.

`Frequency` and `Monetary` are right skewed, so they are passed through `log1p` before scaling. Then `StandardScaler` normalises all three features to zero mean and unit variance, which KMeans needs because it relies on Euclidean distance.

K is selected by silhouette score across `k = 3 .. 6`. The winner is the K with the highest silhouette, not the lowest within cluster sum of squares, because silhouette is more interpretable and penalises clusters that bleed into each other.

Output: `kmeans.joblib` (the model), `rfm_scaler.joblib` (the StandardScaler), `segments.parquet` (per invoice assignment), `segment_profiles.parquet` (per segment averages used by the dashboard)."""

prof_explain = nbformat.v4.new_markdown_cell("Each segment is profiled on size, average Recency, average Quantity, average Monetary, and average Rating. These are the columns the Streamlit Customer Segments page surfaces, so the labels here drive what the business user sees in the dashboard.")
for i, c in enumerate(nb.cells):
    if c.cell_type == 'markdown' and c.source.startswith('## Profile'):
        nb.cells.insert(i + 1, prof_explain)
        break

nbformat.write(nb, 'notebooks/03_customer_segmentation.ipynb')
print('nb03 patched')

# ===== nb04 forecasting =====
nb = nbformat.read('notebooks/04_sales_forecasting.ipynb', as_version=4)
nb.cells[0].source = """# 04 Sales Forecasting

Forecast daily revenue per branch for the next 30 days. Two models compete on each branch and the winner is chosen on out of sample error.

| Model | Why it is here |
|-------|----------------|
| Prophet | Handles weekly seasonality and trend shifts out of the box. Good default for retail daily revenue. |
| ARIMA(2,1,2) | Classical baseline. If Prophet does not beat it, the seasonality structure is probably weak and the simpler model wins on parsimony. |

Validation uses **walk forward cross validation** with three folds of seven days each. Walk forward respects the time order: the model is always trained on the past and scored on the immediate future, never on shuffled rows. The selection metric is mean absolute error (MAE) in USD because it is in the same units as the forecast and easy to communicate.

After scoring, the winning model per branch is refit on the full series and used to project 30 days ahead with an 80 percent confidence band. Prophet models are pickled to `models/prophet_branch_{a,b,c}.joblib` so the dashboard can call `predict` directly without retraining.

Note on Prophet on Windows: Prophet ships a precompiled Stan model and depends on `cmdstanpy`. The fitting loop is wrapped in a stderr redirect so the notebook does not surface the per chain Stan progress lines, which add no information here."""

wf_explain = nbformat.v4.new_markdown_cell("For each branch, the loop fits Prophet and ARIMA on three rolling training windows, scores each on the next seven days, averages MAE across folds, and picks the lower as winner. The whole block is wrapped in a stderr redirect because cmdstanpy emits per chain progress that adds noise without information.")
for i, c in enumerate(nb.cells):
    if c.cell_type == 'markdown' and c.source.startswith('## Fit and score'):
        nb.cells.insert(i + 1, wf_explain)
        break

nbformat.write(nb, 'notebooks/04_sales_forecasting.ipynb')
print('nb04 patched')

# ===== nb05 rating =====
nb = nbformat.read('notebooks/05_rating_prediction.ipynb', as_version=4)
nb.cells[0].source = """# 05 Rating Prediction

Predict the customer rating an invoice will receive (target `Rating`, scale 4 to 10) from transaction context. The use case is twofold: spot drivers of low ratings, and let the dashboard expose a what if predictor.

Two models are trained on the same train test split (80 / 20, seed 42):

| Model | Role | Why |
|-------|------|-----|
| `Ridge` | Linear baseline | Tells us whether anything beats a simple regularised linear fit. If a tree does not, the signal is weak and we should not over claim. |
| `GradientBoostingRegressor` | Non linear contender | Captures interactions (for example branch by product line by hour) that a linear model misses. |

Both models share a `ColumnTransformer` preprocessor: `StandardScaler` on the seven numeric columns, `OneHotEncoder(handle_unknown=ignore)` on the three categorical columns. Wrapping preprocessor and estimator in a `Pipeline` means the saved `rating_model.joblib` accepts the same raw schema the dashboard form produces, no double serialisation needed.

Reported metrics: MAE and RMSE in rating points. The winner (lower MAE) is also the model passed to SHAP for explanation.

SHAP global beeswarm plot answers which features push rating up or down across the population. The dashboard uses local SHAP per prediction to answer why this score for this customer."""

nbformat.write(nb, 'notebooks/05_rating_prediction.ipynb')
print('nb05 patched')

# ===== nb06 basket =====
nb = nbformat.read('notebooks/06_market_basket.ipynb', as_version=4)
nb.cells[0].source = """# 06 Market Basket Analysis

Apriori finds product co occurrence rules of the form `{A, B} => {C}` with support, confidence, and lift. The classical use case requires multi item invoices.

This dataset records one `Product line` per invoice, so a per invoice basket would be a single item set and Apriori would have nothing to mine. The workaround is to **aggregate by `Branch` and `Date`**: each row of the basket matrix is one branch day, and the columns are 1 if that product line sold that day, 0 otherwise. The resulting rules read as: on a typical branch day, when X and Y sell, Z is also likely to sell. This is enough to drive co location and bundling recommendations even without a true item level basket.

Support thresholds are scanned from 0.20 down to 0.03. The first threshold that produces at least one rule is kept, so output exists even on sparse branches. Rules are filtered by `lift > 1` (positive association) and ranked by lift to surface the strongest co movements first."""

nbformat.write(nb, 'notebooks/06_market_basket.ipynb')
print('nb06 patched')
