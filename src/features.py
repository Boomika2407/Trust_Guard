"""Turn raw transaction columns into the 31 features the model expects."""
from __future__ import annotations

import pandas as pd

from . import config


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip column names so 'Time', 'V1', 'Amount' all work."""
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    return out


def missing_columns(df: pd.DataFrame) -> list[str]:
    """Raw columns required for prediction that are absent from `df`."""
    return [c for c in config.RAW_REQUIRED_COLS if c not in df.columns]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model input from a frame with lower-case time, amount and v1..v28."""
    out = df[config.V_COLS].astype("float64").copy()
    out["hour"] = ((df["time"] // 3600) % 24).astype("float64")
    out["amount_scaled"] = (df["amount"] - config.AMOUNT_MEAN) / config.AMOUNT_STD
    out["time_scaled"] = (df["time"] - config.TIME_MEAN) / config.TIME_STD
    return out[config.FEATURE_COLS]
