"""
Unit tests for FeatureEngineer class.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.feature_engineering import FeatureEngineer
from src.config.config_loader import ConfigLoader


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""
    
    def test_init(self, config):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(config)
        assert engineer.config == config
        assert engineer.feature_config is not None
    
    def test_transform_target(self, config, clean_sample_data):
        """Test target transformation."""
        engineer = FeatureEngineer(config)
        df = engineer._transform_target(clean_sample_data.copy())
        
        assert 'cnt_transformed' in df.columns
        if 'cnt' in df.columns:
            # Check transformation (sqrt)
            np.testing.assert_array_almost_equal(
                df['cnt_transformed'] ** 2,
                df['cnt'],
                decimal=5
            )
    
    def test_create_lag_features(self, config, clean_sample_data):
        """Test lag feature creation."""
        engineer = FeatureEngineer(config)
        
        # Transform target first
        df = engineer._transform_target(clean_sample_data.copy())
        df = df.sort_values('timestamp' if 'timestamp' in df.columns else 'dteday').reset_index(drop=True)
        
        df = engineer._create_lag_features(df)
        
        # Should have lag features
        lag_features = config.get("features.lag_features", [1, 24])
        for lag in lag_features:
            col_name = f'cnt_transformed_lag_{lag}h'
            if 'cnt_transformed' in df.columns:
                assert col_name in df.columns
                
                # Check that lag is correct (first lag values should be NaN)
                if lag < len(df):
                    assert pd.isna(df.loc[lag-1, col_name]) or df.loc[lag-1, col_name] == 0
    
    def test_create_rolling_features(self, config, clean_sample_data):
        """Test rolling feature creation."""
        engineer = FeatureEngineer(config)
        
        df = engineer._transform_target(clean_sample_data.copy())
        df = df.sort_values('timestamp' if 'timestamp' in df.columns else 'dteday').reset_index(drop=True)
        
        df = engineer._create_rolling_features(df)
        
        # Should have rolling features
        rolling_windows = config.get("features.rolling_windows", [3, 24])
        for window in rolling_windows:
            mean_col = f'cnt_transformed_roll_mean_{window}h'
            std_col = f'cnt_transformed_roll_std_{window}h'
            if 'cnt_transformed' in df.columns:
                assert mean_col in df.columns
                assert std_col in df.columns
    
    def test_create_cyclical_features(self, config, clean_sample_data):
        """Test cyclical feature creation."""
        engineer = FeatureEngineer(config)
        df = engineer._create_cyclical_features(clean_sample_data.copy())
        
        cyclical_features = config.get("features.cyclical_features", ["hr"])
        for feature in cyclical_features:
            if feature in df.columns:
                assert f'{feature}_sin' in df.columns
                assert f'{feature}_cos' in df.columns
                
                # Check values are in [-1, 1] range
                assert (df[f'{feature}_sin'] >= -1).all()
                assert (df[f'{feature}_sin'] <= 1).all()
                assert (df[f'{feature}_cos'] >= -1).all()
                assert (df[f'{feature}_cos'] <= 1).all()
    
    def test_create_interaction_features(self, config, clean_sample_data):
        """Test interaction feature creation."""
        engineer = FeatureEngineer(config)
        
        # Manually test interaction creation
        # Since interactions are from config, we'll just verify the method works
        df = engineer._create_interaction_features(clean_sample_data.copy())
        
        # The method should not raise errors
        # If interactions are configured and features exist, they should be created
        if 'hr' in df.columns and 'workingday' in df.columns:
            # Check if interaction was created (depends on config)
            interactions = config.get("features.interactions", [])
            if ["hr", "workingday"] in interactions or ["workingday", "hr"] in interactions:
                assert 'hr_workingday' in df.columns or 'workingday_hr' in df.columns
    
    def test_engineer_features_full_pipeline(self, config, clean_sample_data):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer(config)
        
        df_features = engineer.engineer_features(clean_sample_data.copy())
        
        # Should have more columns than input
        assert df_features.shape[1] > clean_sample_data.shape[1]
        
        # Should have target column
        assert 'cnt' in df_features.columns or 'cnt_transformed' in df_features.columns
        
        # Should not have nulls in most columns (some lags may have NaNs at start)
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        null_counts = df_features[numeric_cols].isnull().sum()
        
        # Count nulls in non-lag features (should be minimal)
        lag_cols = [col for col in numeric_cols if 'lag' in col or 'acceleration' in col]
        non_lag_cols = [col for col in numeric_cols if col not in lag_cols]
        non_lag_nulls = df_features[non_lag_cols].isnull().sum().sum()
        
        # Non-lag features should have minimal nulls (< 1% of rows)
        assert non_lag_nulls < len(df_features) * 0.01, f"Too many nulls in non-lag features: {non_lag_nulls}"

