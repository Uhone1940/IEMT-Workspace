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

## Electricity usage prediction system

Schema:
- Features: `residence_type`, `num_tenants`, `season`, `num_appliances`, `num_occupants`, `size_value`, `size_unit` (m2 or ft2), `solar_kwh`
- Targets: `daily_kwh`, `monthly_kwh`, `hourly_kwh`

Quickstart with synthetic data:

```bash
make elec_synth
make elec_train
make elec_evaluate
make elec_predict
```

Train on your CSV:

```bash
python -m ml_project.src.electricity_train \
  --csv path/to/your_electricity.csv \
  --test-size 0.2 \
  --random-state 42 \
  --output-dir artifacts_electricity
```

Prediction on new rows (features only):

```bash
python -m ml_project.src.electricity_predict \
  --artifacts-dir artifacts_electricity \
  --from-csv path/to/new_rows.csv --head 10
```

CSV expectations:
- Residence types: detached house, town house, apartment, cottage
- Seasons: summer, autumn, winter, spring
- `size_unit`: m2 or ft2 (conversion handled automatically)
- Targets should be present for training; for prediction CSV, include features only.