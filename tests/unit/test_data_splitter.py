"""
Unit tests for DataSplitter class.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_splitter import DataSplitter
from src.config.config_loader import ConfigLoader


class TestDataSplitter:
    """Test suite for DataSplitter."""
    
    def test_init(self, config):
        """Test DataSplitter initialization."""
        splitter = DataSplitter(config)
        assert splitter.train_split == config.get("data.train_split", 0.70)
        assert splitter.val_split == config.get("data.val_split", 0.15)
        assert splitter.test_split == config.get("data.test_split", 0.15)
    
    def test_init_invalid_splits(self, config):
        """Test initialization with invalid splits."""
        # Create a mock config with invalid splits
        class InvalidConfig:
            def get(self, key, default=None):
                if "train_split" in key:
                    return 0.5
                elif "val_split" in key:
                    return 0.3
                elif "test_split" in key:
                    return 0.1  # Sums to 0.9, not 1.0
                return default
        
        invalid_config = InvalidConfig()
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DataSplitter(invalid_config)
    
    def test_split_data_temporal(self, config, clean_sample_data):
        """Test temporal data splitting."""
        splitter = DataSplitter(config)
        
        train_df, val_df, test_df = splitter.split_data(
            clean_sample_data,
            time_col='timestamp',
            sort_by_time=True
        )
        
        # Check split sizes
        total = len(clean_sample_data)
        assert len(train_df) == int(total * splitter.train_split)
        assert len(val_df) == int(total * splitter.val_split)
        assert len(train_df) + len(val_df) + len(test_df) == total
        
        # Check temporal ordering (if timestamp exists)
        if 'timestamp' in train_df.columns:
            assert train_df['timestamp'].max() <= val_df['timestamp'].min()
            assert val_df['timestamp'].max() <= test_df['timestamp'].min()
    
    def test_split_data_no_time_col(self, config, clean_sample_data):
        """Test splitting when no time column is present."""
        splitter = DataSplitter(config)
        
        df_no_time = clean_sample_data.drop(columns=['timestamp', 'dteday'], errors='ignore')
        
        train_df, val_df, test_df = splitter.split_data(df_no_time, time_col=None)
        
        # Should still split correctly
        total = len(df_no_time)
        assert len(train_df) + len(val_df) + len(test_df) == total
    
    def test_split_data_auto_detect_time(self, config, clean_sample_data):
        """Test automatic time column detection."""
        splitter = DataSplitter(config)
        
        train_df, val_df, test_df = splitter.split_data(
            clean_sample_data,
            time_col=None,  # Auto-detect
            sort_by_time=True
        )
        
        # Should detect and use timestamp or dteday
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
    
    def test_split_data_no_overlap(self, config, clean_sample_data):
        """Test that splits don't overlap."""
        splitter = DataSplitter(config)
        
        train_df, val_df, test_df = splitter.split_data(clean_sample_data)
        
        # Check indices don't overlap
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)

