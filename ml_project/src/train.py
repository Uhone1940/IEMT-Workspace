import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any

from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, classification_report

from ml_project.src.data import load_dataset
from ml_project.src.model import build_model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_run_prefix(dataset: str, model_type: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"model_{dataset.lower()}_{model_type.lower()}_{timestamp}"


def train_and_evaluate(
    dataset: str,
    model_type: str,
    test_size: float,
    random_state: int,
    output_dir: str,
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
        name=dataset, test_size=test_size, random_state=random_state
    )

    model = build_model(model_type=model_type, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    ensure_dir(output_dir)
    prefix = generate_run_prefix(dataset, model_type)

    model_path = os.path.join(output_dir, f"{prefix}.joblib")
    dump(model, model_path)

    config: Dict[str, Any] = {
        "dataset": dataset,
        "model_type": model_type,
        "test_size": test_size,
        "random_state": random_state,
        "feature_names": feature_names,
        "target_names": target_names,
        "model_path": model_path,
    }
    config_path = os.path.join(output_dir, f"{prefix}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics = {"accuracy": acc, "f1_macro": f1, "classification_report": report}
    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": model_path,
        "config_path": config_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ML model on a dataset")
    parser.add_argument("--dataset", type=str, default="iris", help="Dataset name (default: iris)")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest"],
        help="Model type",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="artifacts", help="Directory to save artifacts"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_evaluate(
        dataset=args.dataset,
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    print(json.dumps({
        "saved_model": result["model_path"],
        "saved_config": result["config_path"],
        "saved_metrics": result["metrics_path"],
        "metrics": result["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()