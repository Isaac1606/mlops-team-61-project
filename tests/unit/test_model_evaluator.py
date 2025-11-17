"""
Unit tests for ModelEvaluator class.
"""

import pytest
import numpy as np
from src.models.model_evaluator import ModelEvaluator
from src.config.config_loader import ConfigLoader


class TestModelEvaluator:
    """Test suite for ModelEvaluator."""
    
    def test_init(self, config):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(config)
        assert evaluator.config == config
        assert evaluator.target_metrics is not None
    
    def test_evaluate_mae(self, config):
        """Test MAE calculation."""
        evaluator = ModelEvaluator(config)
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = evaluator.evaluate(y_true, y_pred, metrics=["mae"])
        
        assert "mae" in metrics
        assert metrics["mae"] > 0
        assert abs(metrics["mae"] - 0.22) < 0.1  # Approximate MAE
    
    def test_evaluate_rmse(self, config):
        """Test RMSE calculation."""
        evaluator = ModelEvaluator(config)
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = evaluator.evaluate(y_true, y_pred, metrics=["rmse"])
        
        assert "rmse" in metrics
        assert metrics["rmse"] > 0
        assert metrics["rmse"] >= metrics.get("mae", 0)  # RMSE >= MAE
    
    def test_evaluate_r2(self, config):
        """Test R² calculation."""
        evaluator = ModelEvaluator(config)
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = evaluator.evaluate(y_true, y_pred, metrics=["r2"])
        
        assert "r2" in metrics
        assert metrics["r2"] <= 1.0  # R² should be <= 1
        assert metrics["r2"] > 0.5  # Good prediction should have high R²
    
    def test_evaluate_all_metrics(self, config):
        """Test evaluation with all metrics."""
        evaluator = ModelEvaluator(config)
        
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "residuals_mean" in metrics
        assert "residuals_std" in metrics
    
    def test_compare_to_targets(self, config):
        """Test target comparison."""
        evaluator = ModelEvaluator(config)
        
        # Simulate good metrics
        metrics = {
            "mae": 80,
            "rmse": 120,
            "r2": 0.85,
            "mape": 10
        }
        
        comparison = evaluator.compare_to_targets(metrics)
        
        # Should have comparisons for configured targets
        assert len(comparison) > 0
        
        # Each comparison should be (value, meets_target) tuple
        for metric_name, (value, meets_target) in comparison.items():
            assert isinstance(value, (int, float))
            assert isinstance(meets_target, bool)
    
    def test_evaluate_perfect_predictions(self, config):
        """Test evaluation with perfect predictions."""
        evaluator = ModelEvaluator(config)
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = y_true.copy()
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert metrics["mae"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1.0
        assert abs(metrics["residuals_mean"]) < 1e-10

