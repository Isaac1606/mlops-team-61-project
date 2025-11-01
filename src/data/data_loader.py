"""
Data loading utilities.
Handles loading data from various sources with validation.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading data from files with validation and error handling.
    
    This class provides a standardized way to load data throughout the project,
    ensuring consistent data types and handling edge cases.
    """
    
    def __init__(self, paths):
        """
        Initialize data loader.
        
        Args:
            paths: ProjectPaths instance for accessing file paths
        """
        self.paths = paths
        logger.info(f"DataLoader initialized with project root: {paths.project_root}")
    
    def load_raw_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            file_path: Path to raw data file. If None, uses default from config.
        
        Returns:
            DataFrame containing raw data
        
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if file_path is None:
            file_path = self.paths.raw_data_file
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        
        logger.info(f"Loading raw data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def load_processed_data(
        self,
        split: str,
        normalized: bool = False,
        file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load processed data (train/val/test).
        
        Args:
            split: Dataset split ("train", "val", "test", or "validation")
            normalized: Whether to load normalized version
            file_path: Custom file path. If None, uses default from paths.
        
        Returns:
            DataFrame containing processed data
        """
        if file_path is None:
            file_path = self.paths.processed_file(split, normalized)
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed data file not found: {file_path}\n"
                f"Run feature engineering pipeline first."
            )
        
        logger.info(f"Loading {split} data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def save_data(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        create_dirs: bool = True
    ) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Path where to save the file
            create_dirs: Whether to create parent directories if they don't exist
        """
        file_path = Path(file_path)
        
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving data to: {file_path}")
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} rows and {len(df.columns)} columns")

