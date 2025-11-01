"""
Scikit-Learn Pipeline for end-to-end ML workflow.
Implements preprocessing → model training as a single pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from .preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    End-to-end ML pipeline using Scikit-Learn Pipeline.
    
    This class wraps preprocessing and model training into a single
    pipeline object, ensuring consistent transformations throughout
    the ML workflow. This is a best practice for production ML systems.
    
    The pipeline consists of:
    1. DataPreprocessor (scaling, feature selection)
    2. Model (estimator)
    
    Benefits:
    - Prevents data leakage (transformations learned only on train)
    - Easy deployment (single object to serialize)
    - Reproducibility (same transformations applied consistently)
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        preprocessor: Optional[DataPreprocessor] = None,
        preprocessor_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ML pipeline.
        
        Args:
            model: Scikit-Learn estimator (e.g., RandomForestRegressor)
            preprocessor: DataPreprocessor instance. If None, creates default.
            preprocessor_config: Configuration dict for creating preprocessor
        """
        self.model = model
        
        # Create preprocessor if not provided
        if preprocessor is None:
            if preprocessor_config is None:
                preprocessor_config = {
                    "scaler_type": "robust",
                    "exclude_from_scaling": [
                        "holiday", "workingday", "season_", "weathersit_",
                        "weather_quadrant_", "is_peak_hour", "is_commute_window"
                    ]
                }
            preprocessor = DataPreprocessor(**preprocessor_config)
        
        self.preprocessor = preprocessor
        
        # Create Scikit-Learn Pipeline
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])
        
        logger.info("MLPipeline initialized")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features (DataFrame)
            y: Training target (Series)
        """
        logger.info("Fitting ML pipeline...")
        self.pipeline.fit(X, y)
        logger.info("Pipeline fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the pipeline.
        
        Args:
            X: Features to predict on (DataFrame)
        
        Returns:
            Predictions (array)
        """
        return self.pipeline.predict(X)
    
    def get_feature_names(self) -> list:
        """Get feature names after preprocessing."""
        if hasattr(self.preprocessor, 'scale_cols_') and hasattr(self.preprocessor, 'non_scale_cols_'):
            return self.preprocessor.scale_cols_ + self.preprocessor.non_scale_cols_
        return []
    
    @property
    def feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from model (if available).
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        model = self.pipeline.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            feature_names = self.get_feature_names()
            if feature_names:
                return dict(zip(feature_names, model.feature_importances_))
            else:
                return dict(enumerate(model.feature_importances_))
        
        return None

