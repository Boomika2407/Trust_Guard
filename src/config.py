"""Project-wide configuration for TrustGuard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"
DATA_DIR = ROOT / "data"
STYLE_PATH = ROOT / "assets" / "style.css"

MODEL_PATH = MODELS_DIR / "fraud_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"  # written by train.py

# ── Feature schema (must match the order the model was trained on) ────────────
V_COLS = [f"v{i}" for i in range(1, 29)]
FEATURE_COLS = V_COLS + ["hour", "amount_scaled", "time_scaled"]
RAW_REQUIRED_COLS = ["time", "amount"] + V_COLS

# ── Scaling constants (StandardScaler stats fitted on the cleaned dataset) ────
TIME_MEAN, TIME_STD = 94811.07759952, 47480.96421642
AMOUNT_MEAN, AMOUNT_STD = 88.47268731099723, 250.39899584557298

# ── Decision threshold ────────────────────────────────────────────────────────
# 0.5 is the model default. Running `python train.py` tunes the threshold on the
# precision-recall curve and stores it in models/metrics.json, which the app
# then uses automatically.
DEFAULT_THRESHOLD = 0.5

# ── Results reported for the trained model (shown when metrics.json is absent) ─
REPORTED_METRICS = {
    "roc_auc": 0.9725,
    "precision": 0.9383,
    "recall": 0.80,
    "f1": 0.8636,
    "fraud_rate": 0.17,
    "n_transactions": 284_807,
}

MODEL_COMPARISON = [
    {"Model": "Logistic Regression", "F1 (Fraud)": 0.1042, "ROC-AUC": 0.9686},
    {"Model": "Decision Tree", "F1 (Fraud)": 0.3584, "ROC-AUC": 0.8886},
    {"Model": "Random Forest", "F1 (Fraud)": 0.8171, "ROC-AUC": 0.9193},
    {"Model": "XGBoost (selected)", "F1 (Fraud)": 0.8506, "ROC-AUC": 0.9725},
]

# ── Triage rules from the business recommendations ────────────────────────────
RULE_V14_THRESHOLD = -5.0
RULE_V4_THRESHOLD = 5.0
