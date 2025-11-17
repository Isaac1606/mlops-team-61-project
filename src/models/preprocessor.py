"""
Data preprocessing using Scikit-Learn transformers.
Handles scaling and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
import joblib
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesses data using Scikit-Learn compatible transformers.
    
    This class implements the Scikit-Learn transformer interface, making
    it compatible with Pipeline. It handles:
    - Feature scaling (StandardScaler, RobustScaler, MinMaxScaler)
    - Feature exclusion (binary/one-hot encoded features)
    
    Implements fit/transform pattern for proper train/test separation.
    """
    
    def __init__(
        self,
        scaler_type: str = "robust",
        exclude_from_scaling: Optional[List[str]] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ("standard", "robust", "minmax", or "none")
            exclude_from_scaling: List of column name prefixes to exclude from scaling
        """
        self.scaler_type = scaler_type
        self.exclude_from_scaling = exclude_from_scaling or []
        
        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        self.feature_cols_ = None
        self.scale_cols_ = None
        self.non_scale_cols_ = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features (DataFrame)
            y: Target variable (optional, not used)
        
        Returns:
            self (for method chaining)
        """
        logger.info(f"Fitting preprocessor (scaler_type={self.scaler_type})")
        
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Clean infinite values and NaNs before fitting
        X = self._clean_infinite_values(X)
        
        # Identify columns to scale
        self.scale_cols_ = [
            col for col in X.columns
            if not any(exclude in col for exclude in self.exclude_from_scaling)
        ]
        
        self.non_scale_cols_ = [
            col for col in X.columns
            if any(exclude in col for exclude in self.exclude_from_scaling)
        ]
        
        logger.debug(f"Scaling {len(self.scale_cols_)} columns, excluding {len(self.non_scale_cols_)}")
        
        # Fit scaler if applicable
        if self.scaler is not None and len(self.scale_cols_) > 0:
            self.scaler.fit(X[self.scale_cols_])
        
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform (DataFrame)
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Clean infinite values and NaNs before transforming
        X = self._clean_infinite_values(X)
        
        # Transform scaled columns
        if self.scaler is not None and len(self.scale_cols_) > 0:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X[self.scale_cols_]),
                columns=self.scale_cols_,
                index=X.index
            )
            
            # Combine scaled and non-scaled columns
            if len(self.non_scale_cols_) > 0:
                X_result = pd.concat([
                    X_scaled,
                    X[self.non_scale_cols_]
                ], axis=1)
            else:
                X_result = X_scaled
        else:
            X_result = X
        
        # Ensure column order matches fit
        if self.scale_cols_ and self.non_scale_cols_:
            col_order = self.scale_cols_ + self.non_scale_cols_
            X_result = X_result[col_order]
        
        return X_result
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, file_path: str) -> None:
        """Save preprocessor to disk."""
        joblib.dump(self, file_path)
        logger.info(f"Saved preprocessor to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'DataPreprocessor':
        """Load preprocessor from disk."""
        return joblib.load(file_path)
    
    def _clean_infinite_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean infinite values and NaNs from DataFrame.
        
        Replaces inf/-inf with NaN, then fills NaN with column median.
        
        Args:
            X: DataFrame to clean
        
        Returns:
            Cleaned DataFrame
        """
        X_clean = X.copy()
        
        # Check for infinite values
        inf_mask = np.isinf(X_clean.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            n_inf = inf_mask.sum().sum()
            logger.warning(f"Found {n_inf} infinite values. Replacing with NaN and filling with median.")
            
            # Replace inf/-inf with NaN
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with column median (for numeric columns only)
            numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X_clean[col].isna().any():
                    median_val = X_clean[col].median()
                    if pd.isna(median_val):
                        # If median is also NaN, use 0 as fallback
                        median_val = 0.0
                    X_clean[col] = X_clean[col].fillna(median_val)
            
            # If still NaN after fill, replace with 0
            X_clean = X_clean.fillna(0)
        
        return X_clean

