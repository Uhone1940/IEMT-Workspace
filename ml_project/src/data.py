from typing import Tuple, List

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


def load_dataset(
    name: str = "iris",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load a dataset and split into train/test.

    Returns:
      X_train, X_test, y_train, y_test, feature_names, target_names
    """
    dataset_key = name.lower()
    if dataset_key == "iris":
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        target_names = list(data.target_names)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names, target_names