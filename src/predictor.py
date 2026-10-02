"""Model loading, scoring and per-transaction explanations."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from . import config


class FraudPredictor:
    """Thin wrapper around the trained sklearn Pipeline (XGBoost inside)."""

    def __init__(self, model_path: Path = config.MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Place fraud_pipeline.pkl in the models/ folder or run `python train.py`."
            )
        self.pipeline = joblib.load(model_path)
        steps = getattr(self.pipeline, "named_steps", None)
        self.model = list(steps.values())[-1] if steps else self.pipeline

    @property
    def supports_shap(self) -> bool:
        """SHAP contributions need an XGBoost model inside the pipeline."""
        return isinstance(self.model, xgb.XGBClassifier)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of fraud for each row."""
        return self.pipeline.predict_proba(features[config.FEATURE_COLS])[:, 1]

    def predict(self, features: pd.DataFrame, threshold: float) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(int)

    def contributions(self, features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Exact TreeSHAP values (log-odds) per feature, plus the base value.

        Uses XGBoost's native `pred_contribs`, which returns the same values as
        shap.TreeExplainer without depending on the shap package at runtime.
        """
        X = features[config.FEATURE_COLS]
        dmat = xgb.DMatrix(X, feature_names=config.FEATURE_COLS)
        raw = self.model.get_booster().predict(dmat, pred_contribs=True)
        values = pd.DataFrame(raw[:, :-1], columns=config.FEATURE_COLS, index=X.index)
        return values, raw[:, -1]


def load_metrics() -> dict:
    """Metrics from the latest training run, falling back to reported results."""
    metrics = {
        **config.REPORTED_METRICS,
        "threshold": config.DEFAULT_THRESHOLD,
        "source": "reported",
    }
    if config.METRICS_PATH.exists():
        try:
            saved = json.loads(config.METRICS_PATH.read_text())
            metrics.update(saved)
            metrics["source"] = "trained"
        except (json.JSONDecodeError, OSError):
            pass
    return metrics
