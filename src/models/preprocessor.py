"""
Data preprocessing using Scikit-Learn transformers.
Handles scaling and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
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
    - Missing value imputation (NaN handling)
    - Feature scaling (StandardScaler, RobustScaler, MinMaxScaler)
    - Feature exclusion (binary/one-hot encoded features)
    
    Implements fit/transform pattern for proper train/test separation.
    """
    
    def __init__(
        self,
        scaler_type: str = "robust",
        exclude_from_scaling: Optional[List[str]] = None,
        impute_strategy: str = "median"
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ("standard", "robust", "minmax", or "none")
            exclude_from_scaling: List of column name prefixes to exclude from scaling
            impute_strategy: Strategy for imputing missing values ("mean", "median", "most_frequent", or "constant")
        """
        self.scaler_type = scaler_type
        self.exclude_from_scaling = exclude_from_scaling or []
        self.impute_strategy = impute_strategy
        
        # Initialize imputer for NaN handling
        self.imputer = SimpleImputer(strategy=impute_strategy, fill_value=0)
        
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
        self.feature_names_ = None  # Store original column order from fit
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
        
        # Check for NaN values
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in training data. Will impute using {self.impute_strategy}.")
        
        # Identify columns to scale (preserve original order)
        self.scale_cols_ = [
            col for col in X.columns
            if not any(exclude in col for exclude in self.exclude_from_scaling)
        ]
        
        self.non_scale_cols_ = [
            col for col in X.columns
            if any(exclude in col for exclude in self.exclude_from_scaling)
        ]
        
        # Store original column order (critical for inference)
        # This preserves the exact order columns appeared during fit
        self.feature_names_ = list(X.columns)
        
        logger.debug(f"Scaling {len(self.scale_cols_)} columns, excluding {len(self.non_scale_cols_)}")
        
        # Fit imputer on all columns (including both scaled and non-scaled)
        self.imputer.fit(X)
        
        # Fit scaler if applicable (only on columns to scale)
        if self.scaler is not None and len(self.scale_cols_) > 0:
            # First impute, then fit scaler
            X_imputed = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index
            )
            self.scaler.fit(X_imputed[self.scale_cols_])
        
        # feature_names_ already set above during column identification
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
        
        # Step 1: Impute missing values
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Step 2: Transform scaled columns
        if self.scaler is not None and len(self.scale_cols_) > 0:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_imputed[self.scale_cols_]),
                columns=self.scale_cols_,
                index=X_imputed.index
            )
            
            # Combine scaled and non-scaled columns
            if len(self.non_scale_cols_) > 0:
                X_result = pd.concat([
                    X_scaled,
                    X_imputed[self.non_scale_cols_]
                ], axis=1)
            else:
                X_result = X_scaled
        else:
            X_result = X_imputed
        
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

