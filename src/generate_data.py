"""Generate a reproducible synthetic dataset for the bus-delay experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 1500
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "bus_delays.csv"


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    distance = rng.uniform(4, 45, n_samples).round(1)
    stops = rng.integers(3, 31, n_samples)
    hour = rng.integers(5, 23, n_samples)
    rain = np.clip(rng.gamma(shape=1.3, scale=2.2, size=n_samples), 0, 15).round(1)
    traffic = rng.choice(["low", "medium", "high"], n_samples, p=[0.28, 0.47, 0.25])
    day_type = rng.choice(["weekday", "weekend"], n_samples, p=[5 / 7, 2 / 7])
    peak_hour = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))).astype(int)

    traffic_delay = pd.Series(traffic).map({"low": 0.5, "medium": 3.0, "high": 7.0}).to_numpy()
    weekend_effect = np.where(day_type == "weekend", -1.0, 0.7)

    # Transparent relationship used only to create demonstration data.
    delay = (
        0.10 * distance
        + 0.13 * stops
        + 0.55 * rain
        + traffic_delay
        + 3.2 * peak_hour
        + weekend_effect
        + rng.normal(0, 2.2, n_samples)
    )
    delay = np.clip(delay, 0, None).round(1)

    return pd.DataFrame(
        {
            "route_distance_km": distance,
            "number_of_stops": stops,
            "departure_hour": hour,
            "rain_mm": rain,
            "traffic_level": traffic,
            "day_type": day_type,
            "peak_hour": peak_hour,
            "delay_minutes": delay,
        }
    )


def main() -> None:
    data = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data)} rows to {OUTPUT_PATH}")
    print(data.head())


if __name__ == "__main__":
    main()
