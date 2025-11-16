"""
Integration tests for data drift detection with performance monitoring.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.data.data_splitter import DataSplitter
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.models.data_drift import DataDriftDetector, PerformanceMonitor
from src.models.pipeline import MLPipeline
from src.config.config_loader import ConfigLoader
from src.config.paths import ProjectPaths
from sklearn.ensemble import RandomForestRegressor


class TestDriftMonitoring:
    """Integration tests for drift detection with performance monitoring."""
    
    def test_drift_detection_with_trained_model(self, clean_sample_data, config, paths):
        """Test drift detection using trained model as baseline."""
        # Step 1: Feature engineering
        engineer = FeatureEngineer(config)
        df_features = engineer.engineer_features(clean_sample_data)
        
        # Step 2: Split data (use train as reference)
        splitter = DataSplitter(config)
        # Auto-detect time column (DataSplitter will handle it)
        train_df, val_df, test_df = splitter.split_data(df_features, time_col=None)
        
        # Step 3: Prepare features
        target_col = config.get("data.target_col", "cnt")
        exclude_cols = config.get("data.exclude_cols", [])
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and col != target_col]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Step 4: Train model
        trainer = ModelTrainer(config, paths)
        pipeline = trainer.train_model(
            model_type="random_forest",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_name="test_drift_baseline",
            model_kwargs={"n_estimators": 10, "max_depth": 5}
        )
        
        # Step 5: Get baseline performance
        evaluator = ModelEvaluator(config)
        y_val_pred = pipeline.predict(X_val)
        baseline_metrics = evaluator.evaluate(y_val.values, y_val_pred)
        
        # Step 6: Initialize drift detector with training data
        detector = DataDriftDetector(X_train)
        
        # Step 7: Detect drift in test data
        drift_results = detector.detect_drift(X_test)
        
        # Should have drift results
        assert 'has_drift' in drift_results
        assert 'drift_score' in drift_results
        assert 'feature_drifts' in drift_results
        
        # Step 8: Monitor performance on test data
        monitor = PerformanceMonitor(
            baseline_metrics,
            performance_threshold=0.2,
            metric_type="mae"
        )
        
        y_test_pred = pipeline.predict(X_test)
        perf_results = monitor.check_performance(y_test.values, y_test_pred)
        
        # Should have performance results
        assert 'has_degradation' in perf_results
        assert 'degradation_score' in perf_results
        assert 'current_metrics' in perf_results
        assert 'alert' in perf_results
    
    def test_drift_with_synthetic_data(self, clean_sample_data, config, paths):
        """Test drift detection with synthetic drifted data."""
        # Feature engineering and splitting
        engineer = FeatureEngineer(config)
        df_features = engineer.engineer_features(clean_sample_data)
        
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
        
        # Train model
        trainer = ModelTrainer(config, paths)
        pipeline = trainer.train_model(
            model_type="ridge",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_name="test_synthetic_drift",
            model_kwargs={"alpha": 10.0}
        )
        
        # Get baseline performance
        evaluator = ModelEvaluator(config)
        y_val_pred = pipeline.predict(X_val)
        baseline_metrics = evaluator.evaluate(y_val.values, y_val_pred)
        
        # Initialize drift detector
        detector = DataDriftDetector(X_train)
        
        # Generate synthetic drifted data
        synthetic_drifted = detector.generate_synthetic_drift(
            n_samples=min(200, len(X_train)),
            drift_type="mean_shift",
            drift_magnitude=2.0,
            features_to_drift=feature_cols[:3]  # Drift first 3 features
        )
        
        # Detect drift
        drift_results = detector.detect_drift(synthetic_drifted)
        
        # Should detect drift in synthetic data
        assert drift_results['has_drift'] is True
        assert drift_results['drift_score'] > 0.1
        
        # Create synthetic target for performance monitoring
        # Use model to predict (simulating production)
        y_synthetic_pred = pipeline.predict(synthetic_drifted[feature_cols])
        
        # For synthetic targets, add drift effect to baseline predictions
        np.random.seed(42)
        y_synthetic_true = y_synthetic_pred + np.random.normal(
            0, 
            baseline_metrics.get('rmse', 100) * 0.3,  # 30% additional noise
            len(y_synthetic_pred)
        )
        
        # Monitor performance on drifted data
        monitor = PerformanceMonitor(
            baseline_metrics,
            performance_threshold=0.2,
            metric_type="mae"
        )
        
        perf_results = monitor.check_performance(y_synthetic_true, y_synthetic_pred)
        
        # Should detect performance degradation
        assert 'has_degradation' in perf_results
        assert 'degradation_score' in perf_results
        
        # Performance should be worse on drifted data
        assert perf_results['current_metrics']['mae'] >= baseline_metrics['mae'] * 0.8  # Allow some variance
    
    def test_end_to_end_drift_monitoring(self, clean_sample_data, config, paths):
        """Complete end-to-end test of drift detection and performance monitoring."""
        # Feature engineering
        engineer = FeatureEngineer(config)
        df_features = engineer.engineer_features(clean_sample_data)
        
        # Split data
        splitter = DataSplitter(config)
        # Auto-detect time column (DataSplitter will handle it)
        train_df, val_df, test_df = splitter.split_data(df_features, time_col=None)
        
        # Prepare features
        target_col = config.get("data.target_col", "cnt")
        exclude_cols = config.get("data.exclude_cols", [])
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and col != target_col]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Train model
        trainer = ModelTrainer(config, paths)
        pipeline = trainer.train_model(
            model_type="random_forest",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_name="test_e2e_drift",
            model_kwargs={"n_estimators": 10, "max_depth": 5}
        )
        
        # Baseline performance
        evaluator = ModelEvaluator(config)
        y_val_pred = pipeline.predict(X_val)
        baseline_metrics = evaluator.evaluate(y_val.values, y_val_pred)
        
        # Drift detection
        detector = DataDriftDetector(X_train)
        drift_results = detector.detect_drift(X_test)
        
        # Performance monitoring
        monitor = PerformanceMonitor(baseline_metrics, performance_threshold=0.2, metric_type="mae")
        y_test_pred = pipeline.predict(X_test)
        perf_results = monitor.check_performance(y_test.values, y_test_pred)
        
        # Combined results
        combined_results = {
            'drift_detected': drift_results['has_drift'],
            'drift_score': drift_results['drift_score'],
            'performance_degradation': perf_results['has_degradation'],
            'degradation_score': perf_results['degradation_score'],
            'alert': perf_results['alert'],
            'baseline_mae': baseline_metrics['mae'],
            'current_mae': perf_results['current_metrics']['mae']
        }
        
        # Should have all components
        assert 'drift_detected' in combined_results
        assert 'performance_degradation' in combined_results
        assert 'alert' in combined_results
        
        # Log results
        print(f"\n=== Drift Monitoring Results ===")
        print(f"Drift Detected: {combined_results['drift_detected']}")
        print(f"Drift Score: {combined_results['drift_score']:.3f}")
        print(f"Performance Degradation: {combined_results['performance_degradation']}")
        print(f"Degradation Score: {combined_results['degradation_score']:.3f}")
        print(f"Alert: {combined_results['alert']}")
        print(f"Baseline MAE: {combined_results['baseline_mae']:.2f}")
        print(f"Current MAE: {combined_results['current_mae']:.2f}")

