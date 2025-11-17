"""
Unit tests for DataCleaner class.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_cleaner import DataCleaner
from src.config.config_loader import ConfigLoader


class TestDataCleaner:
    """Test suite for DataCleaner."""
    
    def test_init(self, config):
        """Test DataCleaner initialization."""
        cleaner = DataCleaner(config)
        assert cleaner.config == config
        assert isinstance(cleaner.exclude_cols, list)
    
    def test_convert_dtypes(self, config, sample_data):
        """Test data type conversion."""
        cleaner = DataCleaner(config)
        df = cleaner._convert_dtypes(sample_data.copy())
        
        # Check date conversion
        if 'dteday' in df.columns:
            assert pd.api.types.is_datetime64_any_dtype(df['dteday']) or \
                   pd.api.types.is_object_dtype(df['dteday'])
        
        # Check numeric conversion
        numeric_cols = ['season', 'hr', 'temp', 'cnt']
        for col in numeric_cols:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col])
    
    def test_handle_nulls(self, config, sample_data):
        """Test null value handling."""
        cleaner = DataCleaner(config)
        
        # Create data with nulls
        df = sample_data.copy()
        df.loc[0:10, 'temp'] = np.nan
        df.loc[5:15, 'hum'] = np.nan
        
        df_clean = cleaner._handle_nulls(df)
        
        # Should have no nulls after cleaning
        assert df_clean.isnull().sum().sum() == 0
    
    def test_remove_problematic_columns(self, config, sample_data):
        """Test removal of problematic columns."""
        cleaner = DataCleaner(config)
        
        df = sample_data.copy()
        df['instant'] = range(len(df))
        df['mixed_type_col'] = 'test'
        
        df_clean = cleaner._remove_problematic_columns(df)
        
        # Should not contain instant
        assert 'instant' not in df_clean.columns
        assert 'mixed_type_col' not in df_clean.columns
    
    def test_validate_cleaned_data(self, config, sample_data):
        """Test data validation."""
        cleaner = DataCleaner(config)
        
        # Valid data should pass
        df_clean = sample_data.copy()
        df_clean = cleaner.clean_data(df_clean)
        cleaner._validate_cleaned_data(df_clean)
        
        # Data with nulls should fail
        df_with_nulls = df_clean.copy()
        df_with_nulls.loc[0, 'temp'] = np.nan
        with pytest.raises(ValueError, match="null values"):
            cleaner._validate_cleaned_data(df_with_nulls)
        
        # Data without target should fail
        df_no_target = df_clean.copy()
        df_no_target = df_no_target.drop(columns=['cnt'])
        with pytest.raises(ValueError, match="Target column"):
            cleaner._validate_cleaned_data(df_no_target)
    
    def test_clean_data_full_pipeline(self, config, sample_data):
        """Test complete cleaning pipeline."""
        cleaner = DataCleaner(config)
        
        # Add some problematic data
        df = sample_data.copy()
        df.loc[0:10, 'temp'] = np.nan
        df['instant'] = range(len(df))
        
        df_clean = cleaner.clean_data(df)
        
        # Should be clean
        assert df_clean.isnull().sum().sum() == 0
        assert 'instant' not in df_clean.columns
        assert 'cnt' in df_clean.columns
        assert len(df_clean) > 0

