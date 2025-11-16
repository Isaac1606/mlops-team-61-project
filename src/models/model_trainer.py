"""
Model training utilities.
Handles model training with MLflow integration.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from typing import Dict, Any, Optional, Union
import mlflow
import mlflow.sklearn
import joblib
import logging

# Setup logger first
logger = logging.getLogger(__name__)

# Optional XGBoost import (handles missing OpenMP runtime gracefully)
try:
    from xgboost import XGBRegressor
    import mlflow.xgboost
    XGBOOST_AVAILABLE = True
except (ImportError, OSError) as e:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None
    logger.warning(
        f"XGBoost not available: {e}\n"
        "To install OpenMP runtime on macOS: brew install libomp\n"
        "XGBoost models will be skipped."
    )

from .pipeline import MLPipeline
from .preprocessor import DataPreprocessor


class ModelTrainer:
    """
    Handles model training with MLflow experiment tracking.
    
    This class encapsulates all model training logic and integrates
    with MLflow for experiment tracking, model versioning, and registry.
    
    Supports:
    - Multiple model types (Ridge, Random Forest, XGBoost)
    - Hyperparameter tuning
    - MLflow experiment tracking
    - Model registry
    """
    
    def __init__(self, config, paths, mlflow_config: Optional[Dict[str, Any]] = None):
        """
        Initialize model trainer.
        
        Args:
            config: ConfigLoader instance
            paths: ProjectPaths instance
            mlflow_config: MLflow configuration (optional, uses config if None)
        """
        self.config = config
        self.paths = paths
        
        # Setup MLflow
        if mlflow_config is None:
            mlflow_config = config.get_section("mlflow")
        
        self.mlflow_config = mlflow_config
        self._setup_mlflow()
        
        logger.info("ModelTrainer initialized")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        tracking_uri = self.mlflow_config.get("tracking_uri", "file:///mlruns")
        
        # Make path absolute if relative
        if tracking_uri.startswith("file:///"):
            tracking_uri = tracking_uri.replace("file:///", "")
            tracking_uri = str(self.paths.project_root / tracking_uri)
            tracking_uri = f"file://{tracking_uri}"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment_name = self.mlflow_config.get("experiment_name", "bike-sharing-demand")
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                tags=self.mlflow_config.get("tags", {})
            )
            logger.info(f"Created MLflow experiment: {experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id if experiment else None
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
    
    def create_model(self, model_type: str, **kwargs) -> Union[Ridge, RandomForestRegressor]:
        """
        Create a model instance from configuration.
        
        Args:
            model_type: Type of model ("ridge", "random_forest", "xgboost")
            **kwargs: Additional hyperparameters (overrides config)
        
        Returns:
            Model instance
        
        Raises:
            ValueError: If model_type is unknown or XGBoost is requested but unavailable
        """
        model_config = self.config.get_section("models")
        
        if model_type.lower() == "ridge":
            params = model_config.get("ridge", {}).copy()
            params.update(kwargs)
            return Ridge(**params)
        
        elif model_type.lower() == "random_forest":
            params = model_config.get("random_forest", {}).copy()
            params.update(kwargs)
            return RandomForestRegressor(**params)
        
        elif model_type.lower() == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ValueError(
                    "XGBoost is not available. Install OpenMP runtime:\n"
                    "  macOS: brew install libomp\n"
                    "  Or reinstall xgboost: pip install --upgrade xgboost"
                )
            params = model_config.get("xgboost", {}).copy()
            params.update(kwargs)
            return XGBRegressor(**params)
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        run_name: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        preprocessor_config: Optional[Dict[str, Any]] = None
    ) -> MLPipeline:
        """
        Train a model with MLflow tracking.
        
        Args:
            model_type: Type of model ("ridge", "random_forest", "xgboost")
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target (optional)
            run_name: MLflow run name (auto-generated if None)
            model_kwargs: Additional model hyperparameters
            preprocessor_config: Preprocessor configuration
        
        Returns:
            Trained MLPipeline
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        if preprocessor_config is None:
            preproc_config = self.config.get_section("preprocessing")
            preprocessor_config = {
                "scaler_type": preproc_config.get("scaler_type", "robust"),
                "exclude_from_scaling": preproc_config.get("exclude_from_scaling", [])
            }
        
        # Create model
        model = self.create_model(model_type, **model_kwargs)
        
        # Create pipeline
        pipeline = MLPipeline(
            model=model,
            preprocessor_config=preprocessor_config
        )
        
        # MLflow run
        if run_name is None:
            run_name = f"{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params(model_kwargs)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", len(X_train))
            
            # Train pipeline
            logger.info(f"Training {model_type} model...")
            if model_type.lower() == "xgboost" and X_val is not None and y_val is not None:
                # XGBoost needs validation set for early stopping
                pipeline.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            else:
                pipeline.fit(X_train, y_train)
            
            # Get predictions for logging
            y_train_pred = pipeline.predict(X_train)
            
            # Log training metrics
            from .model_evaluator import ModelEvaluator
            evaluator = ModelEvaluator(self.config)
            train_metrics = evaluator.evaluate(y_train, y_train_pred)
            
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            # Validation metrics if provided
            if X_val is not None and y_val is not None:
                y_val_pred = pipeline.predict(X_val)
                val_metrics = evaluator.evaluate(y_val, y_val_pred)
                
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value)
            
            # Log model
            if model_type.lower() == "xgboost" and XGBOOST_AVAILABLE:
                mlflow.xgboost.log_model(pipeline.pipeline.named_steps['model'], "model")
            else:
                mlflow.sklearn.log_model(pipeline.pipeline, "model")
            
            # Tags
            mlflow.set_tags({
                "model_family": self._get_model_family(model_type),
                "trainer": "ModelTrainer"
            })
            
            logger.info(f"Model training complete. MLflow run: {mlflow.active_run().info.run_id}")
        
        return pipeline
    
    def _get_model_family(self, model_type: str) -> str:
        """Get model family name for MLflow tags."""
        if model_type.lower() == "ridge":
            return "linear"
        elif model_type.lower() == "random_forest":
            return "ensemble"
        elif model_type.lower() == "xgboost":
            return "boosting"
        else:
            return "unknown"
    
    def save_model(self, pipeline: MLPipeline, model_name: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            pipeline: Trained MLPipeline
            model_name: Name for the model file
        """
        file_path = self.paths.model_file(model_name)
        joblib.dump(pipeline, file_path)
        logger.info(f"Saved model to {file_path}")
    
    def load_model(self, model_name: str) -> MLPipeline:
        """
        Load model from disk.
        
        Args:
            model_name: Name of the model file
        
        Returns:
            Loaded MLPipeline
        """
        file_path = self.paths.model_file(model_name)
        return joblib.load(file_path)

