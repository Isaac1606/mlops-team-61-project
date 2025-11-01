#!/usr/bin/env python
"""
Data pipeline script: Raw → Clean → Feature Engineering → Train/Val/Test Splits

This script implements the complete data processing pipeline:
1. Load raw data
2. Clean data (type conversion, null handling, etc.)
3. Engineer features (lags, rolling stats, cyclical, interactions, etc.)
4. Split into train/validation/test sets
5. Save processed datasets

Run this script before training models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import project modules
from src.config import ConfigLoader, ProjectPaths
from src.data import DataLoader, DataCleaner, FeatureEngineer, DataSplitter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main data processing pipeline."""
    logger.info("="*70)
    logger.info("DATA PROCESSING PIPELINE")
    logger.info("="*70)
    
    # Load configuration
    config = ConfigLoader()
    paths = ProjectPaths(config)
    
    logger.info(f"Project root: {paths.project_root}")
    logger.info(f"Configuration loaded from: {config.config_path}")
    
    # Initialize components
    data_loader = DataLoader(paths)
    data_cleaner = DataCleaner(config)
    feature_engineer = FeatureEngineer(config)
    data_splitter = DataSplitter(config)
    
    # Step 1: Load raw data
    logger.info("\n" + "="*70)
    logger.info("STEP 1: LOADING RAW DATA")
    logger.info("="*70)
    df_raw = data_loader.load_raw_data()
    
    # Step 2: Clean data
    logger.info("\n" + "="*70)
    logger.info("STEP 2: CLEANING DATA")
    logger.info("="*70)
    df_clean = data_cleaner.clean_data(df_raw)
    
    # Save cleaned data
    data_loader.save_data(df_clean, paths.clean_data_file)
    logger.info(f"Cleaned data saved to: {paths.clean_data_file}")
    
    # Step 3: Engineer features
    logger.info("\n" + "="*70)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("="*70)
    df_features = feature_engineer.engineer_features(df_clean)
    
    # Step 4: Split data temporally
    logger.info("\n" + "="*70)
    logger.info("STEP 4: SPLITTING DATA (TEMPORAL)")
    logger.info("="*70)
    train_df, val_df, test_df = data_splitter.split_data(df_features)
    
    # Step 5: Create historical context features (using only train data)
    logger.info("\n" + "="*70)
    logger.info("STEP 5: CREATING HISTORICAL CONTEXT FEATURES (NO LEAKAGE)")
    logger.info("="*70)
    train_df = feature_engineer.create_historical_context_features(train_df, train_df)
    val_df = feature_engineer.create_historical_context_features(train_df, val_df)
    test_df = feature_engineer.create_historical_context_features(train_df, test_df)
    
    # Step 6: Save processed datasets
    logger.info("\n" + "="*70)
    logger.info("STEP 6: SAVING PROCESSED DATASETS")
    logger.info("="*70)
    
    # Save feature-engineered datasets
    train_path = paths.processed_file("train", normalized=False)
    val_path = paths.processed_file("val", normalized=False)
    test_path = paths.processed_file("test", normalized=False)
    
    data_loader.save_data(train_df, train_path)
    data_loader.save_data(val_df, val_path)
    data_loader.save_data(test_df, test_path)
    
    logger.info(f"Train dataset saved: {train_path} ({len(train_df)} rows)")
    logger.info(f"Validation dataset saved: {val_path} ({len(val_df)} rows)")
    logger.info(f"Test dataset saved: {test_path} ({len(test_df)} rows)")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("DATA PROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Raw data: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    logger.info(f"Cleaned data: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    logger.info(f"Features: {len(df_features.columns)} columns")
    logger.info(f"Train: {len(train_df)} rows, {len(train_df.columns)} columns")
    logger.info(f"Validation: {len(val_df)} rows, {len(val_df.columns)} columns")
    logger.info(f"Test: {len(test_df)} rows, {len(test_df.columns)} columns")
    
    logger.info("\n✅ Data processing pipeline completed successfully!")
    logger.info("Next step: Run 'python src/models/train_model.py' to train models.")


if __name__ == "__main__":
    main()

