"""Train, compare, evaluate, and save a bus-delay regression model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bus_delays.csv"
MODEL_PATH = ROOT / "models" / "bus_delay_model.joblib"
RANDOM_SEED = 42

NUMERIC_FEATURES = [
    "route_distance_km",
    "number_of_stops",
    "departure_hour",
    "rain_mm",
    "peak_hour",
]
CATEGORICAL_FEATURES = ["traffic_level", "day_type"]
TARGET = "delay_minutes"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def evaluate(y_true: pd.Series, predictions) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": mean_squared_error(y_true, predictions) ** 0.5,
        "R2": r2_score(y_true, predictions),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python src/generate_data.py")

    data = pd.read_csv(DATA_PATH)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = data[features]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    best_name = None
    best_pipeline = None
    best_mae = float("inf")

    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}\n")

    for name, model in candidates.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate(y_test, predictions)

        print(name)
        print(f"  MAE : {metrics['MAE']:.2f} minutes")
        print(f"  RMSE: {metrics['RMSE']:.2f} minutes")
        print(f"  R²  : {metrics['R2']:.3f}\n")

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_name = name
            best_pipeline = pipeline

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Best model: {best_name} (MAE={best_mae:.2f} minutes)")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
