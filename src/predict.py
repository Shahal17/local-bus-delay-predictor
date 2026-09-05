"""Make one bus-delay prediction using the saved model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bus_delay_model.joblib"


def is_peak_hour(hour: int) -> int:
    return int(7 <= hour <= 9 or 16 <= hour <= 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict local bus delay in minutes")
    parser.add_argument("--distance", type=float, required=True, help="Route distance in km")
    parser.add_argument("--stops", type=int, required=True, help="Number of stops")
    parser.add_argument("--hour", type=int, required=True, choices=range(0, 24), metavar="0-23")
    parser.add_argument("--rain", type=float, default=0.0, help="Rainfall in mm")
    parser.add_argument("--traffic", choices=["low", "medium", "high"], required=True)
    parser.add_argument("--day", choices=["weekday", "weekend"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run generate_data.py and train.py first.")

    model = joblib.load(MODEL_PATH)
    trip = pd.DataFrame(
        [
            {
                "route_distance_km": args.distance,
                "number_of_stops": args.stops,
                "departure_hour": args.hour,
                "rain_mm": max(args.rain, 0),
                "peak_hour": is_peak_hour(args.hour),
                "traffic_level": args.traffic,
                "day_type": args.day,
            }
        ]
    )

    prediction = max(float(model.predict(trip)[0]), 0.0)
    print(f"Estimated delay: {prediction:.1f} minutes")


if __name__ == "__main__":
    main()
