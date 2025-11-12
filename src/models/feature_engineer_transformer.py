"""Scikit-learn transformer that reproduces FeatureEngineer with stateful history."""

from __future__ import annotations

from collections import deque
from typing import Optional, Deque, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.feature_engineering import FeatureEngineer
from src.config import ConfigLoader


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper around FeatureEngineer that maintains a history buffer for
    target-based features (lags, rolling statistics, volatility).

    Parameters
    ----------
    config : ConfigLoader, optional
        Configuration object; if None a default ConfigLoader is created.
    history_window : int
        Maximum number of recent observations to store (default 200).
    """

    def __init__(self, config: Optional[ConfigLoader] = None, history_window: int = 200):
        self.config = config or ConfigLoader()
        self.history_window = history_window
        self._engineer: Optional[FeatureEngineer] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_window)
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the transformer. Stores historical context from training data."""
        self._engineer = FeatureEngineer(self.config)
        # We clone only columns needed to compute historical features
        df = X.copy()
        if y is not None:
            df[self.config.get("data.target_col", "cnt")] = y
        df_eng = self._engineer.engineer_features(df)
        target_col = self.config.get("data.target_col", "cnt")
        for _, row in df_eng[["hr", "weekday", target_col]].iterrows():
            self._history.append(
                {
                    "hr": int(row["hr"]),
                    "weekday": int(row["weekday"]),
                    "cnt": float(row[target_col]),
                    "cnt_transformed": float(np.sqrt(row[target_col])),
                }
            )
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineerTransformer must be fitted before calling transform().")
        if self._engineer is None:
            raise RuntimeError("Internal FeatureEngineer not initialized.")
        df = X.copy()
        df_eng = self._engineer.engineer_features(df)
        target_col = self.config.get("data.target_col", "cnt")
        if target_col in df.columns:
            df_eng[target_col] = df[target_col]
        elif target_col not in df_eng.columns:
            df_eng[target_col] = 0.0
        return df_eng
