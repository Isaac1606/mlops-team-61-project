#!/usr/bin/env python
"""Quick helper to reproduce the basic-feature prediction attempt."""

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ridge_baseline.pkl"

RAW_SAMPLE = {
    "season": [1.0],
    "yr": [0.0],
    "mnth": [1.0],
    "hr": [0.0],
    "holiday": [0.0],
    "weekday": [6.0],
    "workingday": [0.0],
    "weathersit": [1.0],
    "temp": [0.24],
    "atemp": [0.2879],
    "hum": [0.81],
    "windspeed": [0.0],
}


def main() -> None:
    df = pd.DataFrame(RAW_SAMPLE)
    pipeline = joblib.load(MODEL_PATH)
    try:
        preds = pipeline.predict(df)
        print("Prediction:", preds)
    except Exception as exc:
        print("Prediction failed:", exc)


if __name__ == "__main__":
    main()
