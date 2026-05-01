from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "synthetic_qfactor_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLRUNS_DIR = ROOT / "mlruns"

for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

FEATURE_COLS = ["OSNR", "Launch_Power", "Fiber_Length", "Dispersion", "Nonlinear_Effect"]
TARGET_COL = "Q_Factor"

SPLIT_RATIOS = (0.70, 0.15, 0.15)

MLFLOW_URI = f"file:///{MLRUNS_DIR.as_posix()}"
EXPERIMENT_NAME = "qfactor_prediction"
