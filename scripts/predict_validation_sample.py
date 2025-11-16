#!/usr/bin/env python
"""
Run predictions on a sample of validation rows using the best trained model.

The script:
1. Identifies the best model stored in models/.best_model (fallback to available artifacts)
2. Loads the corresponding pipeline
3. Samples 15 rows from the validation dataset
4. Runs predictions and reports input features, predicted cnt, and actual cnt
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.config import ConfigLoader, ProjectPaths
from src.models import ModelTrainer, MLPipeline


SAMPLE_SIZE = 15


def load_best_model(paths: ProjectPaths, config: ConfigLoader) -> Tuple[MLPipeline, str]:
    best_model_file = paths.models_dir / ".best_model"
    trainer = ModelTrainer(config, paths)

    if best_model_file.exists():
        name = best_model_file.read_text().strip()
        if name:
            return trainer.load_model(name), name

    # fallback to available models
    candidates: List[str] = []
    for model_file in paths.models_dir.glob("*.pkl"):
        name = model_file.stem
        if name.endswith("_feature_importance") or name == "scaler":
            continue
        candidates.append(name)

    candidates.sort()
    if not candidates:
        raise RuntimeError("No model artifacts found in models/ directory.")

    name = candidates[0]
    return trainer.load_model(name), name


def sample_validation_data(paths: ProjectPaths) -> pd.DataFrame:
    val_path = paths.processed_file("val", normalized=False)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    df_val = pd.read_csv(val_path)
    if df_val.empty:
        raise ValueError("Validation dataframe is empty.")

    sample_indices = random.sample(range(len(df_val)), k=min(SAMPLE_SIZE, len(df_val)))
    return df_val.iloc[sample_indices].reset_index(drop=True)


def main() -> None:
    random.seed(42)

    config = ConfigLoader()
    paths = ProjectPaths(config)

    print(f"Project root: {paths.project_root}")
    print("Loading best model...")
    pipeline, model_name = load_best_model(paths, config)
    print(f"Using model: {model_name}")

    print("Sampling validation data...")
    df_sample = sample_validation_data(paths)

    feature_cols = [c for c in df_sample.columns if c not in {"cnt", "casual", "registered"}]
    X = df_sample[feature_cols]
    y_true = df_sample["cnt"]

    print("Running predictions...")
    y_pred = pipeline.predict(X)

    report_rows = []
    for idx, (input_row, pred, actual) in enumerate(zip(X.itertuples(index=False), y_pred, y_true)):
        row_dict = input_row._asdict()
        row_dict["prediction_cnt"] = float(pred)
        row_dict["actual_cnt"] = float(actual)
        report_rows.append(row_dict)

    print("\nPrediction Report (15 validation rows):")
    print(json.dumps(report_rows, indent=2))


if __name__ == "__main__":
    main()

