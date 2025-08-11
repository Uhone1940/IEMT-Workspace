import argparse
import json
import os
from datetime import datetime, UTC
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump

from ml_project.src.electricity_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    NUMERIC_MODEL_COLUMNS,
    CATEGORICAL_MODEL_COLUMNS,
    split_features_targets,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp_str() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train electricity usage prediction model")
    p.add_argument("--csv", type=str, required=True, help="Path to training CSV file")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="artifacts_electricity")
    p.add_argument("--n-estimators", type=int, default=400)
    return p.parse_args()


def build_pipeline(n_estimators: int, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_MODEL_COLUMNS,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_MODEL_COLUMNS,
            ),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    return pipe


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    df = pd.read_csv(args.csv)
    X, y = split_features_targets(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipeline = build_pipeline(args.n_estimators, args.random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    # y_pred shape (n_samples, 3)
    metrics: Dict[str, Any] = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        pred_col = y_pred[:, idx]
        true_col = y_test[target].to_numpy()
        mae = mean_absolute_error(true_col, pred_col)
        mse = mean_squared_error(true_col, pred_col)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(true_col, pred_col)
        metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2}

    run_id = f"electricity_{timestamp_str()}"
    model_path = os.path.join(args.output_dir, f"{run_id}.joblib")
    dump(pipeline, model_path)

    config: Dict[str, Any] = {
        "schema": {
            "features": FEATURE_COLUMNS,
            "targets": TARGET_COLUMNS,
            "categorical": CATEGORICAL_MODEL_COLUMNS,
            "numeric": NUMERIC_MODEL_COLUMNS,
        },
        "train_csv": os.path.abspath(args.csv),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": args.n_estimators,
        },
        "model_path": model_path,
    }
    config_path = os.path.join(args.output_dir, f"{run_id}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(args.output_dir, f"{run_id}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({
        "model_path": model_path,
        "config_path": config_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()