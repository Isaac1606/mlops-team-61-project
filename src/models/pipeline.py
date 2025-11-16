"""
Scikit-Learn Pipeline for end-to-end ML workflow.
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from .preprocessor import DataPreprocessor
from .feature_engineer_transformer import FeatureEngineerTransformer

logger = logging.getLogger(__name__)


class MLPipeline:
    def __init__(
        self,
        model: BaseEstimator,
        feature_engineer: Optional[FeatureEngineerTransformer] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        preprocessor_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.feature_engineer = feature_engineer or FeatureEngineerTransformer()

        if preprocessor is None:
            if preprocessor_config is None:
                preprocessor_config = {
                    "scaler_type": "robust",
                    "exclude_from_scaling": [
                        "holiday",
                        "workingday",
                        "season_",
                        "weathersit_",
                        "weather_quadrant_",
                        "is_peak_hour",
                        "is_commute_window",
                    ],
                }
            preprocessor = DataPreprocessor(**preprocessor_config)
        self.preprocessor = preprocessor

        self.pipeline = Pipeline(
            [
                ("feature_engineer", self.feature_engineer),
                ("preprocessor", self.preprocessor),
                ("model", self.model),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        model = self.pipeline.named_steps["model"]

        if hasattr(model, "early_stopping_rounds") and model.early_stopping_rounds is not None:
            if X_val is None or y_val is None:
                raise ValueError("XGBoost with early stopping requires validation data.")
            if not getattr(self.feature_engineer, "_fitted", False):
                self.feature_engineer.fit(X.copy(), y)
            X_train_fe = self.feature_engineer.transform(X.copy())
            X_val_fe = self.feature_engineer.transform(X_val.copy())

            self.preprocessor.fit(X_train_fe, y)
            model.fit(
                self.preprocessor.transform(X_train_fe),
                y,
                eval_set=[
                    (self.preprocessor.transform(X_train_fe), y),
                    (self.preprocessor.transform(X_val_fe), y_val),
                ],
                verbose=False,
            )
        else:
            self.pipeline.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_copy = X.copy()
        if "context_key" in X_copy.columns:
            ctx_series = X_copy.pop("context_key")
            ctx = ctx_series.iloc[0] if len(ctx_series) > 0 else None
        else:
            ctx = None
        model_id = getattr(self.model, "__class__", type(self.model)).__name__.lower()
        X_fe = self.feature_engineer.transform(
            X_copy, model_identifier=model_id, context_key=(ctx if ctx is not None else None)
        )

        expected_order = getattr(self.preprocessor, "feature_names_", None)
        if expected_order:
            missing = [col for col in expected_order if col not in X_fe.columns]
            if missing:
                raise ValueError(f"Missing features for prediction: {missing}")
            extra = [col for col in X_fe.columns if col not in expected_order]
            if extra:
                X_fe = X_fe.drop(columns=extra)
            X_fe = X_fe[expected_order]

        return self.pipeline.named_steps["model"].predict(self.preprocessor.transform(X_fe))

    def get_feature_names(self):
        if hasattr(self.preprocessor, "feature_names_") and self.preprocessor.feature_names_:
            return self.preprocessor.feature_names_
        if hasattr(self.preprocessor, "scale_cols_") and hasattr(self.preprocessor, "non_scale_cols_"):
            return self.preprocessor.scale_cols_ + self.preprocessor.non_scale_cols_
        return []

    @property
    def feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return feature importance scores as a dictionary keyed by feature name.
        Supports tree-based estimators with `feature_importances_` and linear models with `coef_`.
        """
        model = self.pipeline.named_steps.get("model", self.model)

        importances: Optional[np.ndarray] = None
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coef = getattr(model, "coef_")
            if coef is not None:
                importances = np.abs(np.ravel(coef))

        if importances is None:
            return None

        feature_names = self.get_feature_names()
        if feature_names and len(feature_names) == len(importances):
            return {name: float(score) for name, score in zip(feature_names, importances)}

        # Fallback to positional keys when feature names are unavailable/mismatched
        return {f"feature_{idx}": float(score) for idx, score in enumerate(importances)}

