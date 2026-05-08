from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_CSV = DATA_RAW / "supermarket_sales.csv"
TRANSACTIONS_PARQUET = DATA_PROCESSED / "transactions.parquet"
SEGMENTS_PARQUET = DATA_PROCESSED / "segments.parquet"
SEGMENT_PROFILES_PARQUET = DATA_PROCESSED / "segment_profiles.parquet"
FORECAST_PARQUET = DATA_PROCESSED / "forecast.parquet"
RULES_PARQUET = DATA_PROCESSED / "rules.parquet"
RATING_METRICS_PARQUET = DATA_PROCESSED / "rating_metrics.parquet"
RATING_SHAP_PARQUET = DATA_PROCESSED / "rating_shap.parquet"

KMEANS_MODEL = MODELS_DIR / "kmeans.joblib"
SCALER_MODEL = MODELS_DIR / "rfm_scaler.joblib"
RATING_MODEL = MODELS_DIR / "rating_model.joblib"
RATING_PREPROCESSOR = MODELS_DIR / "rating_preprocessor.joblib"
PROPHET_MODEL = lambda branch: MODELS_DIR / f"prophet_branch_{branch.lower()}.joblib"

RANDOM_SEED = 42
BRANCHES = ["A", "B", "C"]
CITY_BY_BRANCH = {"A": "Yangon", "B": "Mandalay", "C": "Naypyitaw"}
PRODUCT_LINES = [
    "Health and beauty",
    "Electronic accessories",
    "Home and lifestyle",
    "Sports and travel",
    "Food and beverages",
    "Fashion accessories",
]
PAYMENT_METHODS = ["Cash", "Credit card", "Ewallet"]
