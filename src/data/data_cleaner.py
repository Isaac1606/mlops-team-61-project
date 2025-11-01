"""
Data cleaning utilities.
Handles data cleaning operations including type conversion, null handling, and outlier removal.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations.
    
    This class encapsulates all data cleaning logic, making it reusable
    and testable. Follows single responsibility principle.
    """
    
    def __init__(self, config):
        """
        Initialize data cleaner.
        
        Args:
            config: ConfigLoader instance for accessing configuration
        """
        self.config = config
        self.exclude_cols = config.get("data.exclude_cols", [])
        logger.info("DataCleaner initialized")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.
        
        This method orchestrates all cleaning steps:
        1. Convert data types
        2. Handle null values
        3. Remove problematic columns
        4. Validate data integrity
        
        Args:
            df: Raw DataFrame to clean
        
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
        
        df_clean = df.copy()
        
        # Step 1: Convert data types
        df_clean = self._convert_dtypes(df_clean)
        
        # Step 2: Handle null values
        df_clean = self._handle_nulls(df_clean)
        
        # Step 3: Remove problematic columns
        df_clean = self._remove_problematic_columns(df_clean)
        
        # Step 4: Validate
        self._validate_cleaned_data(df_clean)
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        
        return df_clean
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        logger.debug("Converting data types...")
        
        # Convert date column if present
        if 'dteday' in df.columns:
            df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Ensure numeric columns are numeric
        numeric_cols = ['season', 'yr', 'mnth', 'hr', 'weekday', 'holiday',
                        'workingday', 'weathersit', 'temp', 'atemp', 'hum',
                        'windspeed', 'casual', 'registered', 'cnt']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle null values.
        
        Strategy:
        - For temporal data: forward fill (carry last known value)
        - For other numeric: forward fill then backward fill
        """
        logger.debug(f"Handling null values. Initial nulls: {df.isnull().sum().sum()}")
        
        # Forward fill temporal data
        temporal_cols = ['temp', 'atemp', 'hum', 'windspeed', 'weathersit']
        for col in temporal_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # For remaining nulls, use forward fill then backward fill
        df = df.ffill().bfill()
        
        # Final check: if any nulls remain, drop those rows
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with remaining nulls")
        
        logger.debug(f"Final nulls: {df.isnull().sum().sum()}")
        
        return df
    
    def _remove_problematic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that are problematic or cause data leakage.
        
        Columns to remove:
        - instant: Row index (not informative)
        - Columns specified in config exclude_cols (after feature engineering)
        """
        logger.debug("Removing problematic columns...")
        
        # Always remove instant if present
        cols_to_remove = ['instant']
        
        # Remove mixed_type_col if present (common issue)
        if 'mixed_type_col' in df.columns:
            cols_to_remove.append('mixed_type_col')
        
        # Remove columns that are in exclude list and exist
        for col in self.exclude_cols:
            if col in df.columns and col not in ['cnt', 'casual', 'registered']:
                # Don't remove target columns at cleaning stage
                cols_to_remove.append(col)
        
        # Remove duplicates
        cols_to_remove = list(set(cols_to_remove))
        
        existing_cols_to_remove = [col for col in cols_to_remove if col in df.columns]
        
        if existing_cols_to_remove:
            logger.debug(f"Removing columns: {existing_cols_to_remove}")
            df = df.drop(columns=existing_cols_to_remove)
        
        return df
    
    def _validate_cleaned_data(self, df: pd.DataFrame) -> None:
        """Validate that cleaned data meets quality standards."""
        # Check for nulls
        if df.isnull().sum().sum() > 0:
            raise ValueError("Cleaned data still contains null values")
        
        # Check for target column
        target_col = self.config.get("data.target_col", "cnt")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in cleaned data")
        
        # Check for reasonable number of rows
        if len(df) < 100:
            raise ValueError(f"Too few rows in cleaned data: {len(df)}")
        
        logger.debug("Data validation passed")

