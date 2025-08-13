import argparse
import glob
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml_project.src.electricity_features import TARGET_COLUMNS, split_features_targets


def find_latest_model(artifacts_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(artifacts_dir, "electricity_*.joblib")))
    return paths[-1] if paths else None


def infer_config_path(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return f"{base}_config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate electricity usage model")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--artifacts-dir", type=str, default="artifacts_electricity")
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

    df = pd.read_csv(config["train_csv"])
    X, y = split_features_targets(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"]
    )

    model = load(model_path)
    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        pred_col = y_pred[:, idx]
        true_col = y_test[target].to_numpy()
        mae = mean_absolute_error(true_col, pred_col)
        mse = mean_squared_error(true_col, pred_col)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(true_col, pred_col)
        metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2}

    print(json.dumps({"model_path": model_path, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()