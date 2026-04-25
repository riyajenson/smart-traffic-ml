from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FEATURE_SETS


def generate_synthetic_data(n_samples: int = 900, seed: int = 42, window_seconds: float = 15.0) -> pd.DataFrame:
    """
    Generate synthetic but realistic traffic data for three congestion levels.

    Columns:
      avg_speed, vehicle_count, speed_variance, inter_arrival_avg, flow_rate, label
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 3
    rows = []

    # LOW congestion
    for _ in range(n):
        avg_speed = rng.normal(220, 30)
        vehicle_count = int(rng.integers(1, 4))
        speed_variance = rng.uniform(50, 400)
        inter_arrival_avg = rng.uniform(4.5, 10.0)
        flow_rate = vehicle_count / window_seconds
        rows.append([max(avg_speed, 155), vehicle_count, speed_variance, inter_arrival_avg, flow_rate, "LOW"])

    # MEDIUM congestion
    for _ in range(n):
        avg_speed = rng.normal(105, 25)
        vehicle_count = int(rng.integers(4, 9))
        speed_variance = rng.uniform(400, 1500)
        inter_arrival_avg = rng.uniform(2.0, 4.0)
        flow_rate = vehicle_count / window_seconds
        rows.append([np.clip(avg_speed, 60, 150), vehicle_count, speed_variance, inter_arrival_avg, flow_rate, "MEDIUM"])

    # HIGH congestion
    for _ in range(n):
        avg_speed = rng.normal(35, 12)
        vehicle_count = int(rng.integers(8, 20))
        speed_variance = rng.uniform(1500, 4000)
        inter_arrival_avg = rng.uniform(0.5, 2.0)
        flow_rate = vehicle_count / window_seconds
        rows.append([max(avg_speed, 5), vehicle_count, speed_variance, inter_arrival_avg, flow_rate, "HIGH"])

    df = pd.DataFrame(
        rows,
        columns=["avg_speed", "vehicle_count", "speed_variance", "inter_arrival_avg", "flow_rate", "label"],
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass(frozen=True)
class CsvLoadResult:
    df: pd.DataFrame
    dropped_rows: int


def load_labeled_csv(csv_path: str) -> CsvLoadResult:
    """
    Load a labeled CSV containing required features + label.
    Keeps only labels in {LOW, MEDIUM, HIGH}.
    """
    df = pd.read_csv(csv_path)

    before = len(df)
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(["LOW", "MEDIUM", "HIGH"])]
    after = len(df)

    # Keep only columns we need (extra columns are ignored but preserved if present)
    required_features = set().union(*FEATURE_SETS.values())
    required = set(required_features) | {"label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    return CsvLoadResult(df=df, dropped_rows=before - after)

