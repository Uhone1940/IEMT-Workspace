import argparse
import glob
import json
import os
from typing import Optional, List

import numpy as np
import pandas as pd
from joblib import load

from ml_project.src.data import load_dataset


def find_latest_model(artifacts_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(artifacts_dir, "model_*.joblib")))
    return paths[-1] if paths else None


def infer_config_path(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return f"{base}_config.json".replace(".joblib_config", "_config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions with a trained model")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to a saved .joblib model"
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default="artifacts", help="Artifacts directory"
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="Path to CSV with feature columns matching the training feature names",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="When no CSV is provided, number of test samples to predict",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model_path or find_latest_model(args.artifacts_dir)
    if not model_path or not os.path.exists(model_path):
        raise SystemExit("No model found. Train a model first or provide --model-path.")

    config_path = infer_config_path(model_path)
    if not os.path.exists(config_path):
        raise SystemExit(f"Config not found for model: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    feature_names: List[str] = config["feature_names"]

    model = load(model_path)

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise SystemExit(
                f"CSV missing required feature columns: {missing}. Expected: {feature_names}"
            )
        X = df[feature_names].to_numpy()
        preds = model.predict(X)
        print(json.dumps({"predictions": preds.tolist()}, indent=2))
    else:
        # Predict on a few samples from the test split
        _, X_test, _, y_test, feature_names, target_names = load_dataset(
            name=config["dataset"],
            test_size=config["test_size"],
            random_state=config["random_state"],
        )
        head = max(1, int(args.head))
        X = X_test[:head]
        preds = model.predict(X)
        print(json.dumps({
            "samples": head,
            "predictions": preds.tolist(),
            "target_names": target_names,
        }, indent=2))


if __name__ == "__main__":
    main()