"""Training utilities for synthetic demand prediction.

Functions provided:
- generate_synthetic_data(n_samples)
- train_and_save_model(df, model_path)
- load_model(model_path)
- predict_from_input(model, input_df)

This module keeps things simple so you can replace data generation with real data later.
"""
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DEFAULT_MODEL_PATH = os.path.join("models", "model.joblib")


def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic dataset that mimics demand patterns.

    Features:
    - hour (0-23)
    - day_of_week (0-6)
    - is_weekend (0/1)
    - weather (0..1) lower is better
    - temperature (C)
    - stations_nearby (int)
    - population_density
    - promotion (0/1)
    - prev_demand (demand in previous hour)

    Target: demand (float)
    """
    rng = np.random.default_rng(random_state)
    hour = rng.integers(0, 24, size=n_samples)
    day = rng.integers(0, 7, size=n_samples)
    is_weekend = (day >= 5).astype(int)
    weather = rng.random(n_samples)  # 0 (good) .. 1 (bad)
    temp = rng.normal(18, 8, size=n_samples)
    stations = rng.integers(0, 50, size=n_samples)
    population_density = rng.integers(500, 15000, size=n_samples)
    promotion = rng.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    # previous demand baseline depends on hour and day
    base = (
        20 +
        30 * np.sin((hour / 24) * 2 * np.pi) +
        10 * (1 - weather) +
        0.0005 * population_density +
        5 * stations +
        15 * promotion -
        10 * is_weekend
    )

    noise = rng.normal(0, 10, size=n_samples)
    prev = np.maximum(0, base + rng.normal(0, 5, size=n_samples))
    demand = np.maximum(0, base * 0.8 + 0.5 * prev + noise)

    df = pd.DataFrame({
        "hour": hour,
        "day_of_week": day,
        "is_weekend": is_weekend,
        "weather": weather,
        "temperature": temp,
        "stations_nearby": stations,
        "population_density": population_density,
        "promotion": promotion,
        "prev_demand": prev,
        "demand": demand,
    })

    return df


def train_and_save_model(df: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH, random_state: int = 42) -> Tuple[object, Dict]:
    """Train an ensemble model on the provided DataFrame and save it to disk.

    Returns: (model, metrics)
    """
    features = [c for c in df.columns if c != "demand"]
    X = df[features]
    y = df["demand"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    ensemble = VotingRegressor(estimators=[("rf", rf), ("gb", gb)])

    ensemble.fit(X_train, y_train)

    pred = ensemble.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    metrics = {"rmse": rmse, "r2": r2}

    # ensure directory exists
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({"model": ensemble, "features": features}, model_path)

    return ensemble, metrics


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load model saved by train_and_save_model. Returns None if not present."""
    if not os.path.exists(model_path):
        return None
    obj = joblib.load(model_path)
    return obj.get("model") if isinstance(obj, dict) else obj


def predict_from_input(model, input_df: pd.DataFrame):
    """Predict using the trained model. Expects input_df to have same feature columns used during training.

    If the saved object included the feature order, caller should pass columns accordingly. Here we assume input_df columns match.
    """
    return model.predict(input_df)


if __name__ == "__main__":
    # quick smoke test when run directly
    df = generate_synthetic_data(500)
    model, metrics = train_and_save_model(df)
    print("Trained model metrics:", metrics)


def load_dataset_from_csv(path: str) -> pd.DataFrame:
    """Load a dataset from CSV and perform basic validation.

    Expects the CSV to contain the same columns used in the synthetic generator:
    `hour,day_of_week,is_weekend,weather,temperature,stations_nearby,population_density,promotion,prev_demand,demand`

    Returns a DataFrame.
    """
    df = pd.read_csv(path)

    expected = {
        "hour",
        "day_of_week",
        "is_weekend",
        "weather",
        "temperature",
        "stations_nearby",
        "population_density",
        "promotion",
        "prev_demand",
        "demand",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Basic type casting
    df = df.loc[:, list(expected)].copy()
    return df
