"""
Integration tests for end-to-end ML pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.data.data_splitter import DataSplitter
from src.models.pipeline import MLPipeline
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.config.config_loader import ConfigLoader
from src.config.paths import ProjectPaths


class TestEndToEndPipeline:
    """Integration tests for complete ML pipeline."""
    
    def test_full_pipeline_ridge(self, sample_data, config, paths):
        """Test complete pipeline with Ridge regression."""
        # Step 1: Clean data
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean_data(sample_data)
        assert len(df_clean) > 0
        
        # Step 2: Feature engineering
        engineer = FeatureEngineer(config)
        df_features = engineer.engineer_features(df_clean)
        assert df_features.shape[1] > df_clean.shape[1]
        
        # Step 3: Split data
        splitter = DataSplitter(config)
        # Auto-detect time column (DataSplitter will handle it)
        train_df, val_df, test_df = splitter.split_data(df_features, time_col=None)
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        
        # Step 4: Prepare features and target
        target_col = config.get("data.target_col", "cnt")
        exclude_cols = config.get("data.exclude_cols", [])
        
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and col != target_col]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        # Step 5: Train model
        trainer = ModelTrainer(config, paths)
        pipeline = trainer.train_model(
            model_type="ridge",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_name="test_ridge_integration"
        )
        
        assert pipeline is not None
        
        # Step 6: Evaluate
        evaluator = ModelEvaluator(config)
        y_pred = pipeline.predict(X_val)
        metrics = evaluator.evaluate(y_val.values, y_pred)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1.0
    
    def test_full_pipeline_random_forest(self, sample_data, config, paths):
        """Test complete pipeline with Random Forest."""
        # Data preparation
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean_data(sample_data)
        
        engineer = FeatureEngineer(config)
        df_features = engineer.engineer_features(df_clean)
        
        splitter = DataSplitter(config)
        # Auto-detect time column (DataSplitter will handle it)
        train_df, val_df, _ = splitter.split_data(df_features, time_col=None)
        
        # Prepare features
        target_col = config.get("data.target_col", "cnt")
        exclude_cols = config.get("data.exclude_cols", [])
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and col != target_col]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        # Train and evaluate
        trainer = ModelTrainer(config, paths)
        pipeline = trainer.train_model(
            model_type="random_forest",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_name="test_rf_integration",
            model_kwargs={"n_estimators": 10, "max_depth": 5}  # Fast for testing
        )
        
        # Check feature importance
        importance = pipeline.feature_importance
        assert importance is not None
        assert len(importance) > 0
        
        # Evaluate
        y_pred = pipeline.predict(X_val)
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate(y_val.values, y_pred)
        
        assert metrics["r2"] > 0.5  # Should have decent R²

