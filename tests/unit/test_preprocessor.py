"""
Unit tests for DataPreprocessor class.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor."""
    
    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor(scaler_type="robust")
        assert preprocessor.scaler_type == "robust"
        assert preprocessor.scaler is not None
    
    def test_init_different_scalers(self):
        """Test initialization with different scalers."""
        # Robust scaler
        robust = DataPreprocessor(scaler_type="robust")
        assert robust.scaler is not None
        
        # Standard scaler
        standard = DataPreprocessor(scaler_type="standard")
        assert standard.scaler is not None
        
        # MinMax scaler
        minmax = DataPreprocessor(scaler_type="minmax")
        assert minmax.scaler is not None
        
        # No scaler
        none = DataPreprocessor(scaler_type="none")
        assert none.scaler is None
    
    def test_init_invalid_scaler(self):
        """Test initialization with invalid scaler type."""
        with pytest.raises(ValueError, match="Unknown scaler_type"):
            DataPreprocessor(scaler_type="invalid")
    
    def test_fit(self, sample_data):
        """Test preprocessor fitting."""
        preprocessor = DataPreprocessor(scaler_type="robust")
        
        # Remove non-numeric columns for fitting
        X = sample_data.select_dtypes(include=[np.number]).drop(columns=['cnt', 'casual', 'registered'], errors='ignore')
        
        preprocessor.fit(X)
        
        assert preprocessor.is_fitted_
        assert preprocessor.scale_cols_ is not None
        assert preprocessor.non_scale_cols_ is not None
    
    def test_fit_exclude_from_scaling(self, sample_data):
        """Test fitting with excluded columns."""
        exclude_list = ["holiday", "workingday"]
        preprocessor = DataPreprocessor(
            scaler_type="robust",
            exclude_from_scaling=exclude_list
        )
        
        X = sample_data.select_dtypes(include=[np.number]).drop(columns=['cnt', 'casual', 'registered'], errors='ignore')
        
        preprocessor.fit(X)
        
        # Excluded columns should not be in scale_cols_
        for exclude in exclude_list:
            if exclude in X.columns:
                assert exclude not in preprocessor.scale_cols_
                assert exclude in preprocessor.non_scale_cols_
    
    def test_transform_before_fit(self, sample_data):
        """Test that transform raises error if not fitted."""
        preprocessor = DataPreprocessor(scaler_type="robust")
        X = sample_data.select_dtypes(include=[np.number]).drop(columns=['cnt', 'casual', 'registered'], errors='ignore')
        
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(X)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor(scaler_type="robust")
        X = sample_data.select_dtypes(include=[np.number]).drop(columns=['cnt', 'casual', 'registered'], errors='ignore')
        
        X_transformed = preprocessor.fit_transform(X)
        
        assert X_transformed.shape == X.shape
        assert preprocessor.is_fitted_
    
    def test_transform_preserves_shape(self, sample_data):
        """Test that transform preserves DataFrame shape."""
        preprocessor = DataPreprocessor(scaler_type="robust")
        X = sample_data.select_dtypes(include=[np.number]).drop(columns=['cnt', 'casual', 'registered'], errors='ignore')
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        assert X_transformed.shape == X.shape
        assert isinstance(X_transformed, pd.DataFrame)

