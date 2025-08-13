import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from ml_project.src.electricity_features import FEATURE_COLUMNS, TARGET_COLUMNS, RESIDENCE_TYPES, SEASONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic electricity dataset")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--output", type=str, default="data/electricity_synth.csv")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    n = int(args.rows)
    residence_type = rng.choice(RESIDENCE_TYPES, size=n)
    season = rng.choice(SEASONS, size=n)

    num_tenants = rng.integers(1, 101, size=n)
    num_occupants = rng.integers(1, 101, size=n)
    num_appliances = rng.integers(1, 101, size=n)

    size_unit = rng.choice(["m2", "ft2"], size=n, p=[0.7, 0.3])
    size_value_m2 = rng.uniform(30, 400, size=n)  # 30-400 m2
    size_value = size_value_m2.copy()
    size_value[size_unit == "ft2"] = size_value_m2[size_unit == "ft2"] / 0.092903

    # Solar generation in kWh per day (roughly 0-60)
    solar_kwh = rng.uniform(0, 60, size=n)

    # Construct a latent daily consumption model (kWh)
    # Base load per m2 and per occupant, appliance contribution, seasonal multiplier
    base_per_m2 = rng.uniform(0.05, 0.12)  # kWh per m2
    base_per_occupant = rng.uniform(1.0, 3.0)  # kWh per occupant
    per_appliance = rng.uniform(0.05, 0.3)  # kWh per appliance

    season_mult = {
        "winter": rng.uniform(1.15, 1.35),
        "summer": rng.uniform(1.05, 1.25),
        "autumn": rng.uniform(0.9, 1.05),
        "spring": rng.uniform(0.9, 1.05),
    }

    size_m2 = size_value_m2

    daily_base = (
        base_per_m2 * size_m2
        + base_per_occupant * num_occupants
        + per_appliance * num_appliances
        + 0.02 * num_tenants
    )

    season_factor = np.array([season_mult[s] for s in season])
    daily_before_solar = daily_base * season_factor

    # Solar offsets consumption up to some fraction
    solar_offset_factor = rng.uniform(0.4, 0.9)
    daily_kwh = np.maximum(0.3, daily_before_solar - solar_offset_factor * solar_kwh)

    # Add noise
    daily_kwh += rng.normal(0, 1.5, size=n)
    daily_kwh = np.maximum(0.2, daily_kwh)

    monthly_kwh = daily_kwh * rng.uniform(29, 31, size=n)
    hourly_kwh = daily_kwh / 24.0 + rng.normal(0, 0.05, size=n)

    df = pd.DataFrame({
        "residence_type": residence_type,
        "num_tenants": num_tenants,
        "season": season,
        "num_appliances": num_appliances,
        "num_occupants": num_occupants,
        "size_value": size_value,
        "size_unit": size_unit,
        "solar_kwh": solar_kwh,
        "daily_kwh": daily_kwh,
        "monthly_kwh": monthly_kwh,
        "hourly_kwh": hourly_kwh,
    })

    ensure_dir(args.output)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()