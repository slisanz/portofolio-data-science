# FTTH Equity Intelligence: Geospatial ML Platform

> Data cleaning pipeline (cp1252 mojibake handling, schema validation) + geospatial layer (H3 hex indexing, WGS84 / Lambert-93 auto-detection, haversine) + composite equity index (3 sub-indicators with rank-percentile normalisation, deterministic snapshot reference) + LightGBM classifier (GroupKFold CV by commune, isotonic calibration, leakage-safe feature set with unit-test guards) + KMeans + PCA commune segmentation + Streamlit dashboard (5 pages, custom theme) + pytest suite (22 tests, anti-leakage guards) + CI workflow.

<a href="https://portofolio-data-science-ftth-equity-intelligence-geospatial.streamlit.app/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"></a>

**Try live demo:** <https://portofolio-data-science-ftth-equity-intelligence-geospatial.streamlit.app/>

Building-level analysis of fibre-to-the-home (FTTH) deployment across Troyes Champagne Métropole, framed as a digital-equity problem. Source: Arcep's open observatory of fixed broadband deployments.

The goal is twofold:

1. Quantify how evenly fibre is being rolled out across the metropolis at the commune and PM (mutualisation point) level.
2. Build a calibrated classifier that flags buildings most likely to lag behind, so a hypothetical operator or local authority could prioritise interventions.

## What is FTTH

Fibre-to-the-home is a fixed-broadband architecture in which the optical fibre runs all the way from the operator's central office to each subscriber's premises, rather than terminating at a street cabinet and switching to copper. In the French market the deployment goes through a hierarchy of access points, most importantly the **Point de Mutualisation (PM)**, a shared aggregation point that serves a few hundred buildings and is what the dataset's `pm_ref` and `pm_etat` columns refer to.

![General architecture of an FTTH network](images/GeneralArchitecture.png)

For this project the relevant unit of analysis is the **building** (`imb_id`), which sits behind a PM, behind an operator's NRO/PRDM. The lag target asks whether a building's *building-level* state (`imb_etat`) has reached `deploye` yet, i.e. whether the last-mile work to actually connect that address has been completed, given that the upstream PM almost always has been.

#### Dashboard demo

**Map Explorer.** Sampled points view (default). Each circle is a building; colour encodes the lag flag. The commune multi-select on top filters down to specific localities; the slider controls the sample size for the points layer.

![Map Explorer, points view](DASHBOARD/Map%20Explorer%20test%201.jpg)

**Map Explorer.** Hex aggregation view. Switch the layer toggle to bin buildings into H3-style hexes for a density read.

![Map Explorer, hex view](DASHBOARD/Map%20Explorer%20test%202.jpg)

**Deployment Predictor.** Calibrated `P(lagging)` for a single building. The page surfaces the holdout AUC / AP / Brier at the top, then lets the reader pick a row from the feature table and edit any of the 16 inputs to see how the predicted probability moves.

![Deployment Predictor, baseline](DASHBOARD/Deployment%20Predictor%20test%201.jpg)
![Deployment Predictor, what-if](DASHBOARD/Deployment%20Predictor%20test%202.jpg)

## Why this project is non-trivial

The interesting part is not the modelling, it's everything around it.

**Data forensics.** The Arcep CSV is encoded in cp1252, mixes float-typed postcodes with NA strings, and has columns named `x`/`y` that look projected but are actually already in WGS84. The cleaning module handles all of that explicitly, with the encoding pinned and the CRS detected from value ranges. There is a dedicated test for each behaviour.

**Pivoting the target on evidence.** The first version of the project defined `is_lagging` on `pm_etat`. After running the EDA we saw that 99.99% of PMs in this snapshot are already deployed; the variance lives at the *building* state (`imb_etat`), where 5.65% of buildings are still in non-deployed states with strong spatial heterogeneity (some communes 0%, others 24%). The target was repointed and the rest of the pipeline updated to match. That kind of decision is exactly what gets discussed at code-review time, and it is documented in-line in the notebooks.

**Anti-leakage by construction.** The feature engineering deliberately excludes any aggregate of the target (no `pm_share_lagging`, no `com_share_lagging`). A unit test in `tests/test_features.py` fails if anyone re-adds them. The reported AUC is consequently lower than a leaky version, and honest.

**Group-aware evaluation.** Cross-validation uses `GroupKFold` on `code_insee`. Buildings in the same commune share latent factors (operator, terrain, deployment plan) so a random k-fold lets the model peek across the train/validation boundary. The CV score is what generalises to a new commune; the random-holdout score is the within-commune signal. Reporting both quantifies how local the predictive power is.

**Honest equity index.** The composite uses three sub-indicators (coverage, PM-load, collective-lag) instead of four. A recency dimension and an operator-competition dimension were considered and **dropped on inspection** because in the Troyes snapshot they collapse to constants: `date_completude` is 100% null and the French RIP model assigns single infrastructure operators per zone (HHI = 1 everywhere). Carrying zero-variance dimensions would dilute the signal; the methodology page explains the choice.

**Calibrated probabilities.** The LightGBM classifier is wrapped in `CalibratedClassifierCV` with isotonic regression so `predict_proba` is interpretable as a frequency, not a score. The notebook plots a reliability diagram to verify.

**Deterministic snapshot.** The recency reference date is the maximum completion date observed in the data, not `datetime.now()`, so re-running the notebooks on a different day produces the same equity scores.

## Data

`data/raw/e-3.csv` contains ~57k buildings with WGS84 coordinates, building category, deployment state, PM reference, and operator code. Source: [data.europa.eu, Arcep observatory, dataset 66cc7372](https://data.europa.eu/data/datasets/66cc737273555682b1e9b051?locale=en). Variable dictionary in `dictionnaire-des-variables.pdf` next to it.

## Layout

```
notebooks/   numbered analysis notebooks, run in order
src/         reusable code (cleaning, geo, features, equity, models, viz)
app/         local Streamlit dashboard (5 pages, dark navy theme)
tests/       pytest suite covering src/
reports/     narrative writeup + exported figures
```

Notebooks write parquet/joblib artefacts to `data/interim/`, `data/processed/`, `models/`. The Streamlit app reads from those locations and degrades gracefully if any are missing; each page tells you which notebook produces it.

## How to run

### Setup

```powershell
# Python 3.11 required (h3 v3 and lightgbm have wheels for 3.11, not 3.14)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Tests

```powershell
python -m pytest -q
```
Should report `22 passed`.

### Notebooks

```powershell
jupyter lab
```

In JupyterLab, run notebooks in this order. For each: **Kernel → Restart Kernel and Run All Cells**.

| # | Notebook | Output |
|---|----------|--------|
| 02 | `02_cleaning_and_geocoding.ipynb` | `data/interim/buildings_clean.parquet` |
| 03 | `03_exploratory_analysis.ipynb` | `reports/figures/headline.json` + 2 PNG |
| 04 | `04_geospatial_analysis.ipynb` | H3 hex stats |
| 05 | `05_equity_index.ipynb` | `data/processed/commune_equity.parquet` |
| 06 | `06_feature_engineering.ipynb` | `data/processed/buildings_features.parquet` |
| 07 | `07_ml_deployment_lag.ipynb` | `models/lag_classifier.joblib` + 3 PNG |
| 08 | `08_clustering_segments.ipynb` | `data/processed/commune_clusters.parquet` + 1 PNG |

Notebook 01 (data audit) and 09 (causal sketch) are exploratory and optional.

### Streamlit dashboard

```powershell
streamlit run app/streamlit_app.py
```

Five pages, ordered so the interactive ones come first: **Map Explorer** (point sample or hex aggregation, commune filter), **Deployment Predictor** (calibrated `P(lagging)` what-if), **Overview** (headline numbers + freshness), **Equity Index** (commune ranking + sub-indicator breakdown), **Methodology** (how each metric is built and what it does *not* say).

The dashboard uses a custom dark theme defined in `.streamlit/config.toml`.

### Re-running from scratch

```powershell
gci data\interim,data\processed,models,reports\figures -File -EA 0 | rm -Force
python -m pytest -q
jupyter lab
```

## Headline metrics (this snapshot)

- 57,072 buildings · 81 communes · 281 PMs · 2 operators
- Building-level lag rate: **5.65%** (range 0% to 24% per commune)
- Classifier: CV AUC **0.74** (group-aware, generalises to unseen communes), holdout AUC **0.91** (within-commune lift), Brier **0.036**
- Equity index range: 0.66 to 0.99 across 81 communes

## Limitations

- One quarterly snapshot, no panel; speed of rollout is not measurable here.
- `date_completude` is 100% null in this file, so no temporal recency analysis.
- Single-operator-per-RIP-zone limits operator-competition analysis to whole-metropolis HHI (≈ 0.54), not per-commune.
- No socioeconomic overlay yet. Joining INSEE filosofi income deciles per IRIS would turn the equity index from *infrastructural* to *outcome*-oriented.
- The causal notebook (09) is observational; do not interpret the uplift estimates as policy-grade.

## License

MIT, see `LICENSE`.
