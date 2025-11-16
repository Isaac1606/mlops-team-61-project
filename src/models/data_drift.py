"""
Data drift detection and performance monitoring.
Detects distribution shifts and performance degradation in production data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detects data drift between training and production data.
    
    Uses statistical tests to identify distribution shifts:
    - Kolmogorov-Smirnov test for continuous features
    - Chi-square test for categorical features
    - PSI (Population Stability Index) for feature-level drift
    
    Also monitors model performance degradation.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        threshold: float = 0.05,
        psi_threshold: float = 0.25
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Training/reference data (baseline distribution)
            feature_columns: List of feature columns to monitor (None = all numeric)
            categorical_columns: List of categorical columns
            threshold: P-value threshold for statistical tests
            psi_threshold: PSI threshold for drift (>= 0.25 indicates significant drift)
        """
        self.reference_data = reference_data.copy()
        self.threshold = threshold
        self.psi_threshold = psi_threshold
        
        # Identify feature columns
        if feature_columns is None:
            # Use all numeric columns by default
            self.feature_columns = list(reference_data.select_dtypes(include=[np.number]).columns)
        else:
            self.feature_columns = feature_columns
        
        # Identify categorical columns
        if categorical_columns is None:
            # Auto-detect categorical (low cardinality numeric or object type)
            self.categorical_columns = []
            for col in reference_data.columns:
                if reference_data[col].dtype == 'object':
                    self.categorical_columns.append(col)
                elif reference_data[col].dtype in ['int64', 'int32'] and \
                     reference_data[col].nunique() < 20:
                    self.categorical_columns.append(col)
        else:
            self.categorical_columns = categorical_columns
        
        # Remove categorical from feature columns if they overlap
        self.feature_columns = [col for col in self.feature_columns 
                               if col not in self.categorical_columns]
        
        # Compute reference statistics
        self.reference_stats = self._compute_statistics(self.reference_data)
        
        logger.info(f"DataDriftDetector initialized with {len(self.feature_columns)} features")
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for reference data."""
        stats_dict = {}
        
        for col in self.feature_columns:
            if col in data.columns:
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75)
                }
        
        for col in self.categorical_columns:
            if col in data.columns:
                stats_dict[col] = {
                    'value_counts': data[col].value_counts(normalize=True).to_dict()
                }
        
        return stats_dict
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        return_details: bool = True
    ) -> Dict[str, Any]:
        """
        Detect drift in current data compared to reference.
        
        Args:
            current_data: Current/production data to check
            return_details: Whether to return detailed drift information
        
        Returns:
            Dictionary with drift detection results:
            - has_drift: bool - Whether drift was detected
            - drift_score: float - Overall drift score (0-1)
            - feature_drifts: dict - Per-feature drift information
            - summary: dict - Summary statistics
        """
        logger.info("Detecting data drift...")
        
        results = {
            'has_drift': False,
            'drift_score': 0.0,
            'feature_drifts': {},
            'summary': {
                'total_features': len(self.feature_columns) + len(self.categorical_columns),
                'drifted_features': 0,
                'tests_performed': 0
            }
        }
        
        drift_scores = []
        
        # Check continuous features
        for col in self.feature_columns:
            if col not in current_data.columns:
                continue
            
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            try:
                ks_statistic, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
                results['summary']['tests_performed'] += 1
                
                # PSI calculation
                psi = self._calculate_psi(ref_values, curr_values)
                
                has_drift = (ks_pvalue < self.threshold) or (psi >= self.psi_threshold)
                
                if has_drift:
                    results['summary']['drifted_features'] += 1
                    results['has_drift'] = True
                
                drift_scores.append(psi)  # Use PSI as drift score
                
                if return_details:
                    results['feature_drifts'][col] = {
                        'type': 'continuous',
                        'has_drift': has_drift,
                        'ks_statistic': float(ks_statistic),
                        'ks_pvalue': float(ks_pvalue),
                        'psi': float(psi),
                        'ref_mean': float(ref_values.mean()),
                        'curr_mean': float(curr_values.mean()),
                        'ref_std': float(ref_values.std()),
                        'curr_std': float(curr_values.std())
                    }
            
            except Exception as e:
                logger.warning(f"Error detecting drift for {col}: {e}")
        
        # Check categorical features
        for col in self.categorical_columns:
            if col not in current_data.columns:
                continue
            
            ref_counts = self.reference_data[col].value_counts(normalize=True)
            curr_counts = current_data[col].value_counts(normalize=True)
            
            # Combine categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            
            if len(all_categories) < 2:
                continue
            
            # Chi-square test
            try:
                # Create contingency table
                ref_freq = [ref_counts.get(cat, 0) * len(self.reference_data) for cat in all_categories]
                curr_freq = [curr_counts.get(cat, 0) * len(current_data) for cat in all_categories]
                
                chi2, chi2_pvalue = stats.chisquare(curr_freq, f_exp=ref_freq)
                results['summary']['tests_performed'] += 1
                
                has_drift = chi2_pvalue < self.threshold
                
                if has_drift:
                    results['summary']['drifted_features'] += 1
                    results['has_drift'] = True
                
                if return_details:
                    results['feature_drifts'][col] = {
                        'type': 'categorical',
                        'has_drift': has_drift,
                        'chi2_statistic': float(chi2),
                        'chi2_pvalue': float(chi2_pvalue),
                        'ref_distribution': ref_counts.to_dict(),
                        'curr_distribution': curr_counts.to_dict()
                    }
            
            except Exception as e:
                logger.warning(f"Error detecting drift for {col}: {e}")
        
        # Overall drift score (average PSI or percentage of drifted features)
        if drift_scores:
            results['drift_score'] = float(np.mean(drift_scores))
        else:
            # Fallback: percentage of drifted features
            total_checked = results['summary']['drifted_features'] + \
                          (results['summary']['tests_performed'] - results['summary']['drifted_features'])
            if total_checked > 0:
                results['drift_score'] = results['summary']['drifted_features'] / total_checked
        
        logger.info(f"Drift detection complete. Drift detected: {results['has_drift']}, "
                   f"Score: {results['drift_score']:.3f}")
        
        return results
    
    def _calculate_psi(self, ref_values: pd.Series, curr_values: pd.Series, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI thresholds:
        - < 0.1: No significant population change
        - 0.1 - 0.25: Moderate population change
        - >= 0.25: Significant population change
        
        Args:
            ref_values: Reference distribution
            curr_values: Current distribution
            bins: Number of bins for discretization
        
        Returns:
            PSI value
        """
        # Create bins based on reference data
        _, bin_edges = np.histogram(ref_values, bins=bins)
        
        # Calculate frequencies
        ref_freq = np.histogram(ref_values, bins=bin_edges)[0]
        curr_freq = np.histogram(curr_values, bins=bin_edges)[0]
        
        # Normalize to probabilities
        ref_prob = ref_freq / len(ref_values)
        curr_prob = curr_freq / len(curr_values)
        
        # Avoid division by zero
        ref_prob = np.where(ref_prob == 0, 1e-10, ref_prob)
        curr_prob = np.where(curr_prob == 0, 1e-10, curr_prob)
        
        # Calculate PSI
        psi = np.sum((curr_prob - ref_prob) * np.log(curr_prob / ref_prob))
        
        return float(psi)
    
    def generate_synthetic_drift(
        self,
        n_samples: int,
        drift_type: str = "mean_shift",
        drift_magnitude: float = 1.0,
        features_to_drift: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data with known drift for testing.
        
        Args:
            n_samples: Number of samples to generate
            drift_type: Type of drift ("mean_shift", "variance_shift", "distribution_shift")
            drift_magnitude: Magnitude of drift (multiplier)
            features_to_drift: List of features to apply drift (None = all)
        
        Returns:
            DataFrame with synthetic drifted data
        """
        logger.info(f"Generating synthetic drift data (type={drift_type}, n={n_samples})")
        
        if features_to_drift is None:
            features_to_drift = self.feature_columns[:3]  # Default: drift first 3 features
        
        synthetic_data = self.reference_data.sample(n=min(n_samples, len(self.reference_data)), 
                                                     replace=True).copy()
        
        for col in features_to_drift:
            if col not in synthetic_data.columns or col not in self.reference_stats:
                continue
            
            ref_stats = self.reference_stats[col]
            
            if drift_type == "mean_shift":
                # Shift mean
                mean_shift = drift_magnitude * ref_stats['std']
                synthetic_data[col] = synthetic_data[col] + mean_shift
            
            elif drift_type == "variance_shift":
                # Increase variance
                variance_multiplier = 1.0 + drift_magnitude
                current_mean = synthetic_data[col].mean()
                synthetic_data[col] = current_mean + \
                    (synthetic_data[col] - current_mean) * np.sqrt(variance_multiplier)
            
            elif drift_type == "distribution_shift":
                # Shift entire distribution (more aggressive)
                mean_shift = drift_magnitude * ref_stats['std']
                synthetic_data[col] = synthetic_data[col] + mean_shift
                # Also add noise
                synthetic_data[col] = synthetic_data[col] + \
                    np.random.normal(0, drift_magnitude * ref_stats['std'] * 0.5, len(synthetic_data))
        
        return synthetic_data


class PerformanceMonitor:
    """
    Monitors model performance degradation over time.
    
    Tracks performance metrics and alerts when performance drops
    below acceptable thresholds.
    """
    
    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        performance_threshold: float = 0.2,
        metric_type: str = "mae"  # "mae", "rmse", or "r2"
    ):
        """
        Initialize performance monitor.
        
        Args:
            baseline_metrics: Dictionary of baseline performance metrics
            performance_threshold: Relative performance degradation threshold (0.2 = 20% worse)
            metric_type: Primary metric to monitor ("mae", "rmse", or "r2")
        """
        self.baseline_metrics = baseline_metrics.copy()
        self.performance_threshold = performance_threshold
        self.metric_type = metric_type.lower()
        
        if self.metric_type not in ["mae", "rmse", "r2"]:
            raise ValueError(f"metric_type must be 'mae', 'rmse', or 'r2', got {self.metric_type}")
        
        logger.info(f"PerformanceMonitor initialized (baseline {self.metric_type}={baseline_metrics.get(self.metric_type, 'N/A')})")
    
    def check_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Check if current performance has degraded.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        
        Returns:
            Dictionary with performance check results:
            - has_degradation: bool - Whether performance degraded
            - degradation_score: float - Relative degradation (0-1+)
            - current_metrics: dict - Current performance metrics
            - baseline_metrics: dict - Baseline metrics
            - alert: bool - Whether to alert (significant degradation)
        """
        # Calculate current metrics
        current_metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred))
        }
        
        baseline_value = self.baseline_metrics.get(self.metric_type)
        current_value = current_metrics[self.metric_type]
        
        if baseline_value is None:
            logger.warning(f"Baseline metric {self.metric_type} not found")
            return {
                'has_degradation': False,
                'degradation_score': 0.0,
                'current_metrics': current_metrics,
                'baseline_metrics': self.baseline_metrics,
                'alert': False
            }
        
        # Calculate relative degradation
        if self.metric_type == "r2":
            # For R², lower is worse
            degradation = (baseline_value - current_value) / abs(baseline_value) if baseline_value != 0 else 0
        else:
            # For MAE/RMSE, higher is worse
            degradation = (current_value - baseline_value) / baseline_value if baseline_value != 0 else 0
        
        has_degradation = degradation > self.performance_threshold
        alert = degradation > (self.performance_threshold * 1.5)  # Alert if significantly worse
        
        results = {
            'has_degradation': has_degradation,
            'degradation_score': float(degradation),
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics,
            'alert': alert
        }
        
        if alert:
            logger.warning(f"Performance degradation detected! {self.metric_type}: "
                         f"{baseline_value:.3f} -> {current_value:.3f} "
                         f"(degradation: {degradation*100:.1f}%)")
        
        return results

