import argparse
import glob
import json
import os
from typing import List, Optional

import pandas as pd
from joblib import load

from ml_project.src.electricity_features import (
    FEATURE_COLUMNS,
    CATEGORICAL_MODEL_COLUMNS,
    NUMERIC_MODEL_COLUMNS,
    TARGET_COLUMNS,
    add_size_m2_column,
    validate_required_columns,
)


def find_latest_model(artifacts_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(artifacts_dir, "electricity_*.joblib")))
    return paths[-1] if paths else None


def infer_config_path(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return f"{base}_config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict electricity usage")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--artifacts-dir", type=str, default="artifacts_electricity")
    p.add_argument("--from-csv", type=str, default=None, help="CSV of feature rows")
    p.add_argument("--head", type=int, default=5, help="Predict first N rows from CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path or find_latest_model(args.artifacts_dir)
    if not model_path or not os.path.exists(model_path):
        raise SystemExit("No model found. Train first or provide --model-path.")

    config_path = infer_config_path(model_path)
    if not os.path.exists(config_path):
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = load(model_path)

    if not args.from_csv:
        raise SystemExit("Provide --from-csv with feature rows to predict.")

    df = pd.read_csv(args.from_csv)
    # Prepare features
    validate_required_columns(df, FEATURE_COLUMNS)
    df = add_size_m2_column(df)
    X = df[CATEGORICAL_MODEL_COLUMNS + NUMERIC_MODEL_COLUMNS]

    n = min(len(X), max(1, int(args.head)))
    preds = model.predict(X.iloc[:n])

    result_rows = []
    for i in range(n):
        row = {"row_index": int(i)}
        row.update({TARGET_COLUMNS[j]: float(preds[i, j]) for j in range(len(TARGET_COLUMNS))})
        result_rows.append(row)

    print(json.dumps({"count": n, "predictions": result_rows}, indent=2))


if __name__ == "__main__":
    main()