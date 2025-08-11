# Minimal ML Project

A small, ready-to-run ML project using scikit-learn on the Iris dataset. Includes training, evaluation, and prediction scripts.

## Quickstart

1. Install dependencies:

```bash
make install
```

2. Train a model (default: logistic regression on Iris):

```bash
make train
```

3. Evaluate the most recent model artifact:

```bash
make evaluate
```

4. Run predictions with the most recent model (on sample data or a CSV):

```bash
make predict
```

## Custom usage

Train with configuration options:

```bash
python -m ml_project.src.train \
  --dataset iris \
  --model logistic \
  --test-size 0.2 \
  --random-state 42 \
  --output-dir artifacts
```

Evaluate a specific model:

```bash
python -m ml_project.src.evaluate --model-path artifacts/model_iris_logistic_<timestamp>.joblib
```

Predict from a CSV (columns must match the training feature names saved in the config):

```bash
python -m ml_project.src.predict \
  --model-path artifacts/model_iris_logistic_<timestamp>.joblib \
  --from-csv path/to/data.csv
```

Artifacts (model, config, metrics) are saved under `artifacts/` by default.