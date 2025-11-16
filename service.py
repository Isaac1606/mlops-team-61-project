#!/usr/bin/env python
"""
FastAPI inference service for Bike Demand Prediction.

Endpoints:
- POST /predict: Run prediction for a single record or a batch of records
- GET  /health:  Health check

Usage:
  uvicorn service:app --host 0.0.0.0 --port 8000 --reload

Note on Historical Features:
  Features requiring historical data (lags, rolling stats) are set to default
  values (0.0) in stateless mode. This is a trade-off for simplicity vs accuracy.
  For production systems, consider implementing a prediction cache to calculate
  real historical features.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple, Iterable, Union

import pandas as pd
import numpy as np
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, StrictFloat, StrictInt, model_validator
from datetime import date, datetime
import hashlib
import json
import time

# Project imports
from src.config import ConfigLoader, ProjectPaths
from src.models import ModelTrainer, MLPipeline
from src.utils.redis_manager import RedisManager

logger = logging.getLogger(__name__)

_CONFIG = ConfigLoader()
_BASE_YEAR = int(_CONFIG.get("features.base_year", 2011))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for startup and shutdown."""
    # Startup: Load the model pipeline
    _load_pipeline()
    yield
    # Shutdown: Cleanup if needed (currently nothing to clean up)


app = FastAPI(title="Bike Demand Predictor", version="1.0.0", lifespan=lifespan)


# ============
# Data models
# ============
class BasicFeatures(BaseModel):
    """Core features expected by the model before derived features are added."""

    dteday: date = Field(..., description="Calendar date (YYYY-MM-DD)")
    temp: StrictFloat = Field(..., ge=0.0, le=1.0, description="Temperature (normalized, 0-1)")
    atemp: StrictFloat = Field(..., ge=0.0, le=1.0, description="Feels-like temperature (normalized, 0-1)")
    hum: StrictFloat = Field(..., ge=0.0, le=1.0, description="Humidity (normalized, 0-1)")
    windspeed: StrictFloat = Field(..., ge=0.0, le=1.0, description="Wind speed (normalized, 0-1)")
    season: StrictInt = Field(..., ge=1, le=4, description="Season (1=spring, 2=summer, 3=fall, 4=winter)")
    weathersit: StrictInt = Field(..., ge=1, le=4, description="Weather situation (1-4)")
    hr: StrictInt = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    weekday: StrictInt = Field(..., ge=0, le=6, description="Weekday (0=Sunday, 6=Saturday)")
    holiday: StrictInt = Field(..., ge=0, le=1, description="Is holiday (0 or 1)")
    workingday: StrictInt = Field(..., ge=0, le=1, description="Is working day (0 or 1)")

    class Config:
        extra = "forbid"


class PredictRecord(BaseModel):
    """Wrapper for batch prediction items with explicit 'features' key."""

    features: BasicFeatures = Field(..., description="Basic feature payload for an individual record")

    class Config:
        extra = "forbid"


class PredictRequest(BaseModel):
    """Unified request schema supporting either single or batch predictions."""

    features: Optional[BasicFeatures] = Field(
        None,
        description="Basic feature values for a single prediction",
    )
    records: Optional[List[Union[BasicFeatures, PredictRecord]]] = Field(
        None,
        description="List of basic feature payloads for batch predictions",
    )

    @model_validator(mode="after")
    def validate_payload(cls, values: "PredictRequest") -> "PredictRequest":
        has_features = values.features is not None
        has_records = values.records is not None and len(values.records) > 0

        if has_features and has_records:
            raise ValueError("Provide either 'features' or 'records', but not both")
        if not has_features and not has_records:
            raise ValueError("Either 'features' or 'records' must be provided")

        return values


class PredictionOutput(BaseModel):
    prediction: float


class PredictionsOutput(BaseModel):
    predictions: List[float]


BASIC_FEATURE_NAMES = [name for name in BasicFeatures.model_fields.keys() if name not in {"yr", "mnth"}]


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status")
    model_loaded: bool = Field(..., description="Indicates whether a model pipeline is loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model currently in memory")
    model_name_source: Optional[str] = Field(None, description="Source that selected the loaded model (env/.best_model/fallback)")
    expected_features: List[str] = Field(..., description="Basic feature names required by the /predict endpoint")
    available_models: List[str] = Field(..., description="List of model artifact names available for inference")


# ===================
# Global state/cache
# ===================
_pipeline: Optional[MLPipeline] = None
_model_name_loaded: Optional[str] = None
_model_name_source: Optional[str] = None
_pipeline_cache: Dict[str, MLPipeline] = {}
_redis_manager = RedisManager()


class PredictionCache:
    """Cache predictions to avoid recomputation for identical inputs."""

    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager

    @staticmethod
    def _normalize_features(features_dict: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(features_dict)
        if "dteday" in normalized and isinstance(normalized["dteday"], (pd.Timestamp, datetime, date)):
            normalized["dteday"] = str(pd.Timestamp(normalized["dteday"]).date())
        return normalized

    def _cache_key(self, features_dict: Dict[str, Any]) -> str:
        normalized = self._normalize_features(features_dict)
        serialized = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, features_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_key = self._cache_key(features_dict)
        payload = self.redis_manager.get_cached_prediction(cache_key)
        if payload is None:
            return None
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception:
            return None

    def store(self, features_dict: Dict[str, Any], prediction_value: float) -> None:
        cache_key = self._cache_key(features_dict)
        payload = json.dumps(
            {
                "features": self._normalize_features(features_dict),
                "prediction": float(prediction_value),
                "timestamp": time.time(),
            }
        ).encode("utf-8")
        self.redis_manager.store_cached_prediction(cache_key, payload)


def _discover_available_models(paths: ProjectPaths) -> List[str]:
    """
    Determine which model artifacts are available for inference based on the latest
    training run metadata. Preference order:
      1. Models listed in `model_comparison.csv` whose `.pkl` artifact exists
      2. Any remaining `.pkl` files in the models directory (excluding helper artifacts)
      3. Fallback to a default list when no artifacts are present
    """
    models_dir = paths.models_dir
    candidates: List[str] = []

    # 1. Use training summary if available
    summary_path = models_dir / "model_comparison.csv"
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            if "model" in summary_df.columns:
                model_names = summary_df["model"].dropna().unique()
                for base_name in model_names:
                    candidate = f"{base_name}_baseline"
                    artifact = models_dir / f"{candidate}.pkl"
                    if artifact.exists():
                        candidates.append(candidate)
                    else:
                        logger.warning(
                            "Model '%s' listed in model_comparison.csv but artifact %s not found",
                            base_name,
                            artifact.name,
                        )
        except Exception as exc:
            logger.warning(
                "Could not read model_comparison.csv to determine available models: %s",
                exc,
            )

    # 2. Fallback to scanning directory for remaining artifacts
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            name = model_file.stem
            if name.endswith("_feature_importance") or name in {"scaler"}:
                continue
            if name not in candidates:
                candidates.append(name)

    # Ensure deterministic ordering
    if not candidates:
        # Fallback to known defaults in case directory is empty
        candidates = ["xgboost_baseline", "random_forest_baseline", "ridge_baseline"]
    else:
        candidates = sorted(set(candidates))

    return candidates


def _get_best_model_name(paths: ProjectPaths) -> tuple[str, str]:
    """
    Get the best model name from .best_model file, or fallback to default.
    
    Priority:
    1. MODEL_NAME environment variable
    2. .best_model file (created by train_model.py)
    3. Default fallback
    """
    # Check environment variable first
    env_model = os.getenv("MODEL_NAME", "").strip()
    if env_model:
        return env_model, "env"
    
    # Try to read best model from file
    best_model_file = paths.models_dir / ".best_model"
    if best_model_file.exists():
        try:
            best_model_name = best_model_file.read_text().strip()
            if best_model_name:
                return best_model_name, ".best_model"
        except Exception:
            pass  # Fall through to default
    
    # Default fallback: first available model
    available = _discover_available_models(paths)
    return available[0], "fallback"


def _load_pipeline(force_reload: bool = False) -> MLPipeline:
    """
    Load the trained pipeline from models directory.
    Model name priority:
    1. MODEL_NAME environment variable
    2. .best_model file (created by train_model.py)
    3. Default: first available model artifact
    """
    config = ConfigLoader()
    paths = ProjectPaths(config)
    desired_model, model_source = _get_best_model_name(paths)
    available_models = _discover_available_models(paths)

    global _pipeline, _model_name_loaded, _model_name_source
    if (
        not force_reload
        and _pipeline is not None
        and _model_name_loaded == desired_model
    ):
        return _pipeline

    trainer = ModelTrainer(config, paths)
    model_name = desired_model
    actual_source = model_source
    try:
        pipeline = trainer.load_model(model_name)
    except FileNotFoundError as e:
        # Fallback: try other available models discovered dynamically
        fallbacks = available_models
        tried = [model_name]
        for alt in fallbacks:
            if alt == model_name:
                continue
            try:
                pipeline = trainer.load_model(alt)
                model_name = alt
                actual_source = f"fallback:{model_source}->{alt}"
                logger.warning(
                    "Model '%s' could not be loaded (%s). Using fallback model '%s'.",
                    desired_model,
                    e,
                    alt,
                )
                break
            except FileNotFoundError:
                tried.append(alt)
                continue
        else:
            raise HTTPException(
                status_code=500,
                detail=f"No se pudo cargar ningún modelo. Intentados: {tried}. Error original: {str(e)}",
            )

    _attach_redis_manager(pipeline)
    _pipeline = pipeline
    _model_name_loaded = model_name
    _model_name_source = actual_source
    _pipeline_cache[model_name] = pipeline
    return pipeline


def _load_pipeline_for_model(model_identifier: str, force_reload: bool = False) -> MLPipeline:
    """
    Load a specific model pipeline by identifier (e.g., 'xgboost_baseline').
    Raises 404 if the artifact is not available.
    """
    if not force_reload and model_identifier in _pipeline_cache:
        return _pipeline_cache[model_identifier]

    config = ConfigLoader()
    paths = ProjectPaths(config)
    trainer = ModelTrainer(config, paths)

    try:
        pipeline = trainer.load_model(model_identifier)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_identifier}' no encontrado. "
                   f"Asegúrate de que el archivo {model_identifier}.pkl exista en la carpeta models/.",
        )

    _attach_redis_manager(pipeline)
    _pipeline_cache[model_identifier] = pipeline
    return pipeline


def _attach_redis_manager(pipeline: MLPipeline) -> None:
    feature_engineer = getattr(pipeline, "feature_engineer", None)
    if feature_engineer is not None:
        feature_engineer.use_redis = True
        feature_engineer.redis_manager = _redis_manager


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from basic input features.
    
    This function automatically calculates:
    - Cyclical encodings (sin/cos for hr, mnth, weekday)
    - Interaction features (temp×hum, temp×windspeed, etc.)
    - Temporal indicators (is_peak_hour, is_commute_window)
    
    **Historical Features Limitation:**
    Features that require historical data (lags, rolling stats, volatility)
    are set to default values (0.0) because they need past target values
    that are not available in a stateless inference service.
    
    For production systems requiring higher accuracy, consider:
    - Maintaining a cache/DB of recent predictions to calculate real lags
    - Requiring clients to provide historical demand values
    
    Args:
        df: DataFrame with basic input features
        
    Returns:
        DataFrame with all derived features (historical ones set to 0.0)
    """
    df = df.copy()
    
    # Cyclical encodings (can be computed from basic features)
    if 'hr' in df.columns:
        hr_max = 23  # Assuming 0-23 hour range
        df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / hr_max)
        df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / hr_max)
    
    if 'mnth' in df.columns:
        mnth_max = 12  # Assuming 1-12 month range
        df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / mnth_max)
        df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / mnth_max)
    
    if 'weekday' in df.columns:
        weekday_max = 6  # Assuming 0-6 weekday range
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / weekday_max)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / weekday_max)
    
    # Interaction features (can be computed)
    if 'temp' in df.columns and 'hum' in df.columns:
        df['temp_hum'] = df['temp'] * df['hum']
        df['temp_hum_interaction'] = df['temp'] * df['hum']
    
    if 'temp' in df.columns and 'windspeed' in df.columns:
        df['temp_windspeed'] = df['temp'] * df['windspeed']
        df['temp_wind_interaction'] = df['temp'] * df['windspeed']
    
    if 'temp' in df.columns:
        df['temp_squared'] = df['temp'] ** 2
    
    # Temporal indicators (can be computed)
    if 'hr' in df.columns:
        df['is_peak_hour'] = ((df['hr'] >= 7) & (df['hr'] <= 9)) | ((df['hr'] >= 17) & (df['hr'] <= 19))
        df['is_peak_hour'] = df['is_peak_hour'].astype(int)
        df['is_commute_window'] = ((df['hr'] >= 6) & (df['hr'] <= 10)) | ((df['hr'] >= 16) & (df['hr'] <= 20))
        df['is_commute_window'] = df['is_commute_window'].astype(int)
    
    if 'hr' in df.columns and 'workingday' in df.columns:
        df['hr_workingday'] = df['hr'] * df['workingday']
    
    # Weather interaction
    if 'weathersit' in df.columns:
        df['is_perfect_weather'] = (df['weathersit'] == 1).astype(int)
    
    # One-hot encoding for season (if not already provided)
    if 'season' in df.columns:
        for i in range(1, 5):
            col_name = f'season_{i}'
            if col_name not in df.columns:
                df[col_name] = (df['season'] == i).astype(int)
    
    # ====================================================================
    # HISTORICAL FEATURES - DEFAULT VALUES
    # ====================================================================
    # These features require past target values (demand history) that are
    # not available in stateless inference. They are set to 0.0 (neutral).
    #
    # Why they can't be calculated:
    # - Lags: Need actual demand from 1h, 24h, 48h, 72h, 168h ago
    # - Rolling stats: Need series of past values (3h, 24h, 72h windows)
    # - Volatility: Need std/mean of past 24h values
    # - Acceleration: Need 2+ past values to compute 2nd derivative
    # - Historical context: cnt_vs_historical needs lag_1h (past value)
    #
    # Impact: Model accuracy may be reduced (~5-15% depending on feature
    # importance). For production, consider implementing a prediction cache.
    # ====================================================================
    historical_features = [
        # Lags: Past target values (require actual demand history)
        'cnt_transformed_lag_1h',      # Demand 1 hour ago
        'cnt_transformed_lag_24h',    # Demand 24 hours ago (same time yesterday)
        'cnt_transformed_lag_48h',    # Demand 48 hours ago
        'cnt_transformed_lag_72h',    # Demand 72 hours ago (3 days)
        'cnt_transformed_lag_168h',   # Demand 168 hours ago (1 week)
        
        # Rolling statistics: Moving averages/std over time windows
        'cnt_transformed_roll_mean_3h',   # Avg demand last 3 hours
        'cnt_transformed_roll_std_3h',    # Std dev last 3 hours
        'cnt_transformed_roll_mean_24h',  # Avg demand last 24 hours
        'cnt_transformed_roll_std_24h',   # Std dev last 24 hours
        'cnt_transformed_roll_mean_72h',  # Avg demand last 72 hours
        'cnt_transformed_roll_std_72h',   # Std dev last 72 hours
        
        # Volatility: Measures of demand variability
        'cnt_cv_24h',              # Coefficient of variation (std/mean) last 24h
        'cnt_volatility_24h',      # Volatility (std of changes) last 24h
        
        # Momentum: Rate of change and acceleration
        'cnt_acceleration_1h',     # 2nd derivative: change of change (1h)
        'cnt_acceleration_24h',    # 2nd derivative: change of change (24h)
        'cnt_pct_change_1h',      # % change vs 1 hour ago
        'cnt_pct_change_24h',      # % change vs 24 hours ago
        
        # Historical context: Comparison with historical patterns
        'cnt_historical_avg_raw',  # Historical average (pre-calculated from training)
        'cnt_vs_historical'        # Difference: lag_1h - historical_avg
    ]
    
    for feat in historical_features:
        if feat not in df.columns:
            df[feat] = 0.0  # Default neutral value (model trained with real values)
    
    return df


def _records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert incoming records to a DataFrame using only the basic features.
    Performs validation and type normalization; the heavy feature engineering
    is delegated to the scikit-learn pipeline (FeatureEngineerTransformer +
    DataPreprocessor) to guarantee parity with training.
    """
    df = pd.DataFrame.from_records(records)

    if df.empty:
        raise HTTPException(status_code=400, detail="No se recibieron columnas de entrada")

    missing_basic = [c for c in BASIC_FEATURE_NAMES if c not in df.columns]
    if missing_basic:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Faltan columnas requeridas: {missing_basic}. "
                f"Proporciona exactamente estas columnas básicas: {BASIC_FEATURE_NAMES}"
            ),
        )

    df = df[BASIC_FEATURE_NAMES].copy()
    df["dteday"] = pd.to_datetime(df["dteday"])

    # Derive temporal indices relative to base year
    dteday_series = df["dteday"]
    df["yr"] = (dteday_series.dt.year - _BASE_YEAR).astype(int)
    df["mnth"] = dteday_series.dt.month.astype(int)

    int_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    float_cols = ["temp", "atemp", "hum", "windspeed"]
    for col in float_cols:
        df[col] = df[col].astype(float)

    ordered_cols = BASIC_FEATURE_NAMES + ["yr", "mnth"]
    df = df.reindex(columns=ordered_cols)

    return df


def _persist_predictions_to_history(
    pipeline: MLPipeline,
    features_df: pd.DataFrame,
    predictions: Iterable[float],
) -> None:
    """
    Store predicted cnt values in Redis so future inferences can use real lags.
    """
    preds_array = np.asarray(list(predictions), dtype=float).flatten()

    for row, pred in zip(features_df.itertuples(index=False), preds_array):
        if np.isnan(pred):
            continue

        dt = pd.Timestamp(row.dteday)
        hour = int(row.hr)
        timestamp = datetime(dt.year, dt.month, dt.day, hour).timestamp()
        cnt_transformed = float(np.sqrt(pred)) if pred >= 0 else 0.0

        try:
            _redis_manager.write_prediction(cnt_transformed, timestamp)
            logger.debug(
                "Persisted prediction to Redis",
                extra={
                    "dteday": dt.date().isoformat(),
                    "hr": hour,
                    "cnt": float(pred),
                    "cnt_transformed": cnt_transformed,
                },
            )
        except Exception as exc:
            logger.warning("No se pudo persistir el historial en Redis: %s", exc)


def _extract_records(payload: PredictRequest) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Normalize a PredictRequest into a list of record dictionaries and flag
    indicating whether it was a single prediction.
    """
    if payload.features is not None:
        return [payload.features.model_dump()], True
    if not payload.records:
        raise HTTPException(status_code=400, detail="La lista 'records' está vacía")

    normalized: List[Dict[str, Any]] = []
    for record in payload.records:
        if isinstance(record, BasicFeatures):
            normalized.append(record.model_dump())
        else:
            normalized.append(record.features.model_dump())
    return normalized, False


@app.get("/health", response_model=HealthResponse)
def health():
    config = ConfigLoader()
    paths = ProjectPaths(config)
    available_models = _discover_available_models(paths)
    required_basic_features = BASIC_FEATURE_NAMES.copy()
    pipeline = _load_pipeline(force_reload=True)
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        model_name=_model_name_loaded,
        model_name_source=_model_name_source,
        expected_features=required_basic_features,
        available_models=available_models,
    )


@app.post("/predict", response_model=PredictionOutput | PredictionsOutput)
def predict(payload: PredictRequest):
    """
    Accepts either:
      - single record: { "features": { "<feature>": value, ... } }
      - batch:         { "records": [ { "<feature>": value, ... }, ... ] }
    """
    pipeline = _load_pipeline()
    cache = PredictionCache(_redis_manager)

    # Normalize input to batch of records
    records, single = _extract_records(payload)

    if single:
        record = dict(records[0])
        cached = cache.get(record)
        if cached is not None:
            logger.debug("Returning cached prediction for features: %s", record)
            return PredictionOutput(prediction=float(cached["prediction"]))

    try:
        X = _records_to_dataframe(records)
        preds = pipeline.predict(X)
        _persist_predictions_to_history(pipeline, X, preds)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(exc)}")

    if single:
        prediction_value = float(preds[0])
        cache.store(record, prediction_value)
        return PredictionOutput(prediction=prediction_value)
    return PredictionsOutput(predictions=[float(p) for p in preds])


@app.post(
    "/models/{model_name}/{model_version}/predict",
    response_model=PredictionOutput | PredictionsOutput,
)
def predict_with_model(model_name: str, model_version: str, payload: PredictRequest):
    """
    Accepts the same payload as /predict but lets clients choose the model
    and version to use for inference. The service expects an artifact named
    `<model_name>_<model_version>.pkl` inside the `models/` directory.
    """
    model_identifier = f"{model_name}_{model_version}"
    pipeline = _load_pipeline_for_model(model_identifier)
    cache = PredictionCache(_redis_manager)
    records, single = _extract_records(payload)

    cache_record = None
    if single:
        cache_record = dict(records[0])
        cache_record["__model"] = model_identifier
        cached = cache.get(cache_record)
        if cached is not None:
            logger.debug("Returning cached prediction for model %s features %s", model_identifier, cache_record)
            return PredictionOutput(prediction=float(cached["prediction"]))

    try:
        X = _records_to_dataframe(records)
        preds = pipeline.predict(X)
        _persist_predictions_to_history(pipeline, X, preds)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al predecir con el modelo '{model_identifier}': {exc}",
        )

    if single:
        prediction_value = float(preds[0])
        if cache_record is not None:
            cache.store(cache_record, prediction_value)
        return PredictionOutput(prediction=prediction_value)
    return PredictionsOutput(predictions=[float(p) for p in preds])


# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)