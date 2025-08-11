import argparse
import glob
import json
import os
from typing import Optional, Tuple

from joblib import load
from sklearn.metrics import accuracy_score, f1_score, classification_report

from ml_project.src.data import load_dataset


def find_latest_model(artifacts_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(artifacts_dir, "model_*.joblib")))
    return paths[-1] if paths else None


def infer_config_path(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return f"{base}_config.json".replace(".joblib_config", "_config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to a saved .joblib model"
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default="artifacts", help="Artifacts directory"
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

    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
        name=config["dataset"],
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    model = load(model_path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    result = {"model_path": model_path, "accuracy": acc, "f1_macro": f1, "classification_report": report}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()