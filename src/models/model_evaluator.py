"""
Model evaluation utilities.
Computes metrics and generates evaluation reports.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles model evaluation and metric computation.
    
    This class provides a standardized way to evaluate models,
    compute multiple metrics, and compare against targets.
    """
    
    def __init__(self, config):
        """
        Initialize model evaluator.
        
        Args:
            config: ConfigLoader instance for target metrics
        """
        self.config = config
        self.target_metrics = config.get_section("evaluation").get("target_metrics", {})
        logger.info("ModelEvaluator initialized")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            metrics: List of metrics to compute. If None, uses config defaults.
        
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = self.config.get("evaluation.metrics", ["mae", "rmse", "r2", "mape"])
        
        results = {}
        
        # MAE
        if "mae" in metrics:
            results["mae"] = float(mean_absolute_error(y_true, y_pred))
        
        # RMSE
        if "rmse" in metrics:
            mse = mean_squared_error(y_true, y_pred)
            results["rmse"] = float(np.sqrt(mse))
        
        # R²
        if "r2" in metrics:
            results["r2"] = float(r2_score(y_true, y_pred))
        
        # MAPE
        if "mape" in metrics:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            results["mape"] = float(mape)
        
        # Additional metrics
        residuals = y_true - y_pred
        results["residuals_mean"] = float(residuals.mean())
        results["residuals_std"] = float(residuals.std())
        
        return results
    
    def compare_to_targets(self, metrics: Dict[str, float]) -> Dict[str, Tuple[float, bool]]:
        """
        Compare computed metrics against target metrics.
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            Dictionary mapping metric names to (value, meets_target) tuples
        """
        comparison = {}
        
        for metric_name, target_value in self.target_metrics.items():
            if metric_name.lower() in metrics:
                computed_value = metrics[metric_name.lower()]
                
                # Determine if target is met (higher is better for R², lower for others)
                if metric_name.lower() == "r2":
                    meets_target = computed_value >= target_value
                else:
                    meets_target = computed_value <= target_value
                
                comparison[metric_name] = (computed_value, meets_target)
        
        return comparison
    
    def print_evaluation_report(
        self,
        metrics: Dict[str, float],
        dataset_name: str = "Validation"
    ) -> None:
        """
        Print formatted evaluation report.
        
        Args:
            metrics: Dictionary of computed metrics
            dataset_name: Name of the dataset being evaluated
        """
        print(f"\n{'='*70}")
        print(f"EVALUATION REPORT - {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Print each metric with target comparison
        comparison = self.compare_to_targets(metrics)
        
        for metric_name, (value, meets_target) in comparison.items():
            target = self.target_metrics[metric_name]
            status = "✓" if meets_target else "✗"
            direction = ">" if metric_name.lower() == "r2" else "<"
            print(f"{metric_name.upper():8s}: {value:10.2f}  {status}  (target: {direction} {target})")
        
        # Print residuals
        print(f"\nResiduals:")
        print(f"  Mean: {metrics.get('residuals_mean', 0):10.2f}  (should be ~0)")
        print(f"  Std:  {metrics.get('residuals_std', 0):10.2f}")
        
        print(f"{'='*70}")
    
    def get_feature_importance(
        self,
        pipeline,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained pipeline.
        
        Args:
            pipeline: Trained MLPipeline
            top_n: Number of top features to return (None for all)
        
        Returns:
            DataFrame with feature names and importance scores
        """
        importance_dict = pipeline.feature_importance
        
        if importance_dict is None:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            df = df.head(top_n)
        
        return df

