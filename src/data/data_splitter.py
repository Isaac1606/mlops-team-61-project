"""
Data splitting utilities.
Handles temporal train/validation/test splits.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Handles temporal data splitting.
    
    For time series data, we must split temporally (not randomly)
    to avoid data leakage and maintain realistic evaluation.
    """
    
    def __init__(self, config):
        """
        Initialize data splitter.
        
        Args:
            config: ConfigLoader instance for accessing split ratios
        """
        self.config = config
        self.train_split = config.get("data.train_split", 0.70)
        self.val_split = config.get("data.val_split", 0.15)
        self.test_split = config.get("data.test_split", 0.15)
        
        # Validate splits sum to 1.0
        total = self.train_split + self.val_split + self.test_split
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(
                f"Splits must sum to 1.0, got {total}: "
                f"train={self.train_split}, val={self.val_split}, test={self.test_split}"
            )
        
        logger.info(f"DataSplitter initialized: train={self.train_split}, val={self.val_split}, test={self.test_split}")
    
    def split_data(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        sort_by_time: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets temporally.
        
        Args:
            df: DataFrame to split
            time_col: Column name for time-based sorting. If None, auto-detects.
            sort_by_time: Whether to sort by time before splitting
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data. Initial shape: {df.shape}")
        
        df_split = df.copy()
        
        # Auto-detect time column
        if time_col is None:
            if 'timestamp' in df_split.columns:
                time_col = 'timestamp'
            elif 'dteday' in df_split.columns:
                time_col = 'dteday'
            else:
                logger.warning("No time column found, splitting by index order")
                time_col = None
        
        # Sort by time if specified
        if sort_by_time and time_col is not None:
            df_split = df_split.sort_values(time_col).reset_index(drop=True)
            logger.debug(f"Sorted by {time_col}")
        
        # Calculate split indices
        n_total = len(df_split)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        
        # Split
        train_df = df_split.iloc[:n_train].copy()
        val_df = df_split.iloc[n_train:n_train + n_val].copy()
        test_df = df_split.iloc[n_train + n_val:].copy()
        
        logger.info(f"Split complete:")
        logger.info(f"  Train: {len(train_df)} rows ({len(train_df)/n_total*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df)} rows ({len(val_df)/n_total*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df)} rows ({len(test_df)/n_total*100:.1f}%)")
        
        # Log time ranges
        if time_col and time_col in train_df.columns:
            logger.info(f"  Train period: {train_df[time_col].min()} to {train_df[time_col].max()}")
            logger.info(f"  Val period:   {val_df[time_col].min()} to {val_df[time_col].max()}")
            logger.info(f"  Test period:  {test_df[time_col].min()} to {test_df[time_col].max()}")
        
        return train_df, val_df, test_df

