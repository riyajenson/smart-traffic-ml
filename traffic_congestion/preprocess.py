from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_SETS, DEFAULT_FEATURE_SET


def preprocess_dataframe(df: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET) -> pd.DataFrame:
    """
    Minimal, Arduino-friendly preprocessing:
    - clip extreme speed outliers at p99 (reduces sensitivity to occasional sensor glitches)
    - fill missing inter-arrival gaps with column mean
    - ensure vehicle_count is int
    - clip all feature values to be non-negative
    """
    df = df.copy()

    # Clip speed outliers at 99th percentile
    p99 = df["avg_speed"].quantile(0.99)
    df["avg_speed"] = df["avg_speed"].clip(upper=p99)

    # Fill missing inter-arrival values with the mean
    df["inter_arrival_avg"] = df["inter_arrival_avg"].fillna(df["inter_arrival_avg"].mean())

    # Ensure vehicle_count is integer
    df["vehicle_count"] = df["vehicle_count"].astype(int)

    # Clip negative values caused by sensor noise
    features = FEATURE_SETS[feature_set]
    for col in features:
        df[col] = df[col].clip(lower=0)

    return df


def build_xy(df: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET) -> tuple[np.ndarray, np.ndarray]:
    features = FEATURE_SETS[feature_set]
    X = df[features].to_numpy()
    y = df["label"].to_numpy()
    return X, y

