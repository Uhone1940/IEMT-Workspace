from __future__ import annotations

from typing import List, Tuple

import pandas as pd

# Canonical category values
RESIDENCE_TYPES: List[str] = [
    "detached house",
    "town house",
    "apartment",
    "cottage",
]
SEASONS: List[str] = ["summer", "autumn", "winter", "spring"]

FEATURE_COLUMNS: List[str] = [
    "residence_type",
    "num_tenants",
    "season",
    "num_appliances",
    "num_occupants",
    "size_value",
    "size_unit",
    "solar_kwh",
]

TARGET_COLUMNS: List[str] = [
    "daily_kwh",
    "monthly_kwh",
    "hourly_kwh",
]

# Processed numeric feature columns used in the model
NUMERIC_MODEL_COLUMNS: List[str] = [
    "num_tenants",
    "num_appliances",
    "num_occupants",
    "size_m2",
    "solar_kwh",
]

CATEGORICAL_MODEL_COLUMNS: List[str] = [
    "residence_type",
    "season",
]


def add_size_m2_column(df: pd.DataFrame) -> pd.DataFrame:
    if "size_m2" in df.columns:
        return df
    size_value = df["size_value"].astype(float)
    unit_series = df["size_unit"].str.strip().str.lower().replace({
        "m2": "m2",
        "m^2": "m2",
        "sqm": "m2",
        "ft2": "ft2",
        "ft^2": "ft2",
        "sqft": "ft2",
    })
    ft2_mask = unit_series == "ft2"
    m2 = size_value.copy()
    m2[ft2_mask] = size_value[ft2_mask] * 0.092903
    df = df.copy()
    df["size_m2"] = m2
    return df


def validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validate_required_columns(df, FEATURE_COLUMNS)
    validate_required_columns(df, TARGET_COLUMNS)
    df = add_size_m2_column(df)
    X = df[CATEGORICAL_MODEL_COLUMNS + NUMERIC_MODEL_COLUMNS]
    y = df[TARGET_COLUMNS]
    return X, y