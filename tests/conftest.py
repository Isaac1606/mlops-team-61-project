"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.config.config_loader import ConfigLoader
from src.config.paths import ProjectPaths


@pytest.fixture(scope="session")
def config():
    """Load configuration for testing."""
    return ConfigLoader()


@pytest.fixture(scope="session")
def paths(config):
    """Create paths instance for testing."""
    return ProjectPaths(config)


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2011-01-01', periods=n_samples, freq='H')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'dteday': dates.date,
        'season': np.random.randint(1, 5, n_samples),
        'yr': np.random.randint(0, 2, n_samples),
        'mnth': np.random.randint(1, 13, n_samples),
        'hr': np.random.randint(0, 24, n_samples),
        'holiday': np.random.randint(0, 2, n_samples),
        'weekday': np.random.randint(0, 7, n_samples),
        'workingday': np.random.randint(0, 2, n_samples),
        'weathersit': np.random.randint(1, 5, n_samples),
        'temp': np.random.uniform(0.0, 1.0, n_samples),
        'atemp': np.random.uniform(0.0, 1.0, n_samples),
        'hum': np.random.uniform(0.0, 1.0, n_samples),
        'windspeed': np.random.uniform(0.0, 1.0, n_samples),
        'casual': np.random.randint(0, 200, n_samples),
        'registered': np.random.randint(0, 500, n_samples),
        'cnt': np.random.randint(0, 700, n_samples)
    })
    
    return df


@pytest.fixture
def clean_sample_data(sample_data):
    """Create cleaned sample data."""
    from src.data.data_cleaner import DataCleaner
    from src.config.config_loader import ConfigLoader
    
    config = ConfigLoader()
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean_data(sample_data)
    
    # Restore timestamp if it was removed (needed for some tests)
    if 'timestamp' not in df_clean.columns and 'timestamp' in sample_data.columns:
        df_clean['timestamp'] = sample_data['timestamp'].iloc[:len(df_clean)]
    
    return df_clean


@pytest.fixture
def train_val_test_data(clean_sample_data):
    """Create train/validation/test splits."""
    from src.data.data_splitter import DataSplitter
    from src.config.config_loader import ConfigLoader
    
    config = ConfigLoader()
    splitter = DataSplitter(config)
    return splitter.split_data(clean_sample_data, time_col='timestamp')


@pytest.fixture
def sample_model():
    """Create a sample trained model."""
    from sklearn.ensemble import RandomForestRegressor
    from src.models.pipeline import MLPipeline
    
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    pipeline = MLPipeline(model=model)
    return pipeline

