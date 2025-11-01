#!/usr/bin/env python
"""
Model training script with MLflow integration.

This script:
1. Loads processed train/validation/test datasets
2. Trains multiple models (Ridge, Random Forest, XGBoost)
3. Evaluates models on validation and test sets
4. Logs all experiments to MLflow
5. Saves trained models and feature importance

Run this script after 'src/data/make_dataset.py' has generated processed datasets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging

# Import project modules
from src.config import ConfigLoader, ProjectPaths
from src.data import DataLoader
from src.models import ModelTrainer, ModelEvaluator, DataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_features_and_target(df: pd.DataFrame, config) -> tuple:
    """
    Prepare features (X) and target (y) from DataFrame.
    
    Args:
        df: DataFrame with features and target
        config: ConfigLoader instance
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Get exclude columns
    exclude_cols = config.get("data.exclude_cols", [])
    
    # Feature columns (all except excluded)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Target column
    target_col = config.get("data.target_col", "cnt")
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y, feature_cols


def main():
    """Main model training pipeline."""
    logger.info("="*70)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("="*70)
    
    # Load configuration
    config = ConfigLoader()
    paths = ProjectPaths(config)
    
    logger.info(f"Project root: {paths.project_root}")
    logger.info(f"Configuration loaded from: {config.config_path}")
    
    # Initialize components
    data_loader = DataLoader(paths)
    model_trainer = ModelTrainer(config, paths)
    evaluator = ModelEvaluator(config)
    
    # Load processed datasets
    logger.info("\n" + "="*70)
    logger.info("LOADING PROCESSED DATASETS")
    logger.info("="*70)
    
    train_df = data_loader.load_processed_data("train", normalized=False)
    val_df = data_loader.load_processed_data("val", normalized=False)
    test_df = data_loader.load_processed_data("test", normalized=False)
    
    # Prepare features and targets
    X_train, y_train, feature_cols = prepare_features_and_target(train_df, config)
    X_val, y_val, _ = prepare_features_and_target(val_df, config)
    X_test, y_test, _ = prepare_features_and_target(test_df, config)
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Train: {X_train.shape}")
    logger.info(f"Validation: {X_val.shape}")
    logger.info(f"Test: {X_test.shape}")
    
    # Train models
    models_to_train = ["ridge", "random_forest", "xgboost"]
    trained_models = {}
    results_summary = []
    
    for model_type in models_to_train:
        logger.info("\n" + "="*70)
        logger.info(f"TRAINING {model_type.upper()} MODEL")
        logger.info("="*70)
        
        try:
            # Train model
            pipeline = model_trainer.train_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                run_name=f"{model_type}_baseline"
            )
            
            trained_models[model_type] = pipeline
            
            # Evaluate on all splits
            logger.info("\nEvaluating model...")
            
            # Train metrics
            y_train_pred = pipeline.predict(X_train)
            train_metrics = evaluator.evaluate(y_train, y_train_pred)
            evaluator.print_evaluation_report(train_metrics, "Train")
            
            # Validation metrics
            y_val_pred = pipeline.predict(X_val)
            val_metrics = evaluator.evaluate(y_val, y_val_pred)
            evaluator.print_evaluation_report(val_metrics, "Validation")
            
            # Test metrics
            y_test_pred = pipeline.predict(X_test)
            test_metrics = evaluator.evaluate(y_test, y_test_pred)
            evaluator.print_evaluation_report(test_metrics, "Test")
            
            # Save model
            model_trainer.save_model(pipeline, f"{model_type}_baseline")
            
            # Save feature importance
            importance_df = evaluator.get_feature_importance(pipeline, top_n=20)
            if not importance_df.empty:
                importance_path = paths.feature_importance_file(model_type)
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"Feature importance saved to: {importance_path}")
            
            # Store results
            results_summary.append({
                'model': model_type,
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_r2': val_metrics['r2'],
                'val_mape': val_metrics['mape'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_r2': test_metrics['r2'],
                'test_mape': test_metrics['mape']
            })
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*70)
    
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = paths.models_dir / "model_comparison.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")
        
        # Identify best model
        best_model = summary_df.loc[summary_df['val_rmse'].idxmin()]
        logger.info(f"\n🏆 Best model (lowest validation RMSE): {best_model['model']}")
        logger.info(f"   Validation RMSE: {best_model['val_rmse']:.2f}")
        logger.info(f"   Test RMSE: {best_model['test_rmse']:.2f}")
    
    logger.info("\n✅ Model training pipeline completed successfully!")
    logger.info("View MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()

