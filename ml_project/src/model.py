from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_model(
    model_type: Literal["logistic", "random_forest"] = "logistic",
    random_state: int = 42,
) -> Pipeline:
    model_key = model_type.lower()

    if model_key == "logistic":
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        multi_class="auto",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_key == "random_forest":
        pipeline = Pipeline(
            steps=[
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200, random_state=random_state
                    ),
                )
            ]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return pipeline