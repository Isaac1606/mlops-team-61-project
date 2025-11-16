"""
Tests for data drift detection and performance monitoring.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.data_drift import DataDriftDetector, PerformanceMonitor


class TestDataDriftDetector:
    """Test suite for DataDriftDetector."""
    
    @pytest.fixture
    def reference_data(self):
        """Create reference data."""
        np.random.seed(42)
        n_samples = 500
        
        return pd.DataFrame({
            'feature1': np.random.normal(10, 2, n_samples),
            'feature2': np.random.normal(5, 1, n_samples),
            'feature3': np.random.uniform(0, 100, n_samples),
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples)
        })
    
    def test_init(self, reference_data):
        """Test DataDriftDetector initialization."""
        detector = DataDriftDetector(reference_data)
        
        assert detector.reference_data is not None
        assert len(detector.feature_columns) > 0
        assert detector.threshold > 0
    
    def test_no_drift(self, reference_data):
        """Test detection when there's no drift."""
        detector = DataDriftDetector(reference_data, threshold=0.05)
        
        # Use same data (should detect no drift)
        current_data = reference_data.copy()
        
        results = detector.detect_drift(current_data)
        
        # Should have minimal drift when using same data
        assert 'has_drift' in results
        assert 'drift_score' in results
        assert results['drift_score'] < 0.25  # Low PSI indicates no drift
    
    def test_mean_shift_drift(self, reference_data):
        """Test detection of mean shift drift."""
        detector = DataDriftDetector(reference_data, threshold=0.05)
        
        # Create drifted data (mean shift)
        current_data = reference_data.copy()
        current_data['feature1'] = current_data['feature1'] + 5  # Significant shift
        
        results = detector.detect_drift(current_data)
        
        # Should detect drift in feature1
        assert results['has_drift'] == True
        assert 'feature1' in results['feature_drifts']
        assert results['feature_drifts']['feature1']['has_drift'] == True
    
    def test_variance_shift_drift(self, reference_data):
        """Test detection of variance shift drift."""
        detector = DataDriftDetector(reference_data, threshold=0.05)
        
        # Create drifted data (variance increase)
        current_data = reference_data.copy()
        mean_val = current_data['feature2'].mean()
        current_data['feature2'] = mean_val + (current_data['feature2'] - mean_val) * 3  # 3x variance
        
        results = detector.detect_drift(current_data)
        
        # Should detect drift
        assert results['has_drift'] is True
        assert results['drift_score'] > 0.1
    
    def test_categorical_drift(self, reference_data):
        """Test detection of categorical drift."""
        detector = DataDriftDetector(reference_data, threshold=0.05)
        
        # Create drifted categorical data
        current_data = reference_data.copy()
        # Change distribution (more of one category)
        current_data['categorical'] = np.random.choice(['A', 'A', 'A', 'B', 'C'], len(current_data))
        
        results = detector.detect_drift(current_data)
        
        # Should detect drift in categorical column
        assert 'categorical' in results['feature_drifts']
        assert results['feature_drifts']['categorical']['type'] == 'categorical'
    
    def test_generate_synthetic_drift_mean_shift(self, reference_data):
        """Test synthetic drift generation with mean shift."""
        detector = DataDriftDetector(reference_data)
        
        synthetic_drifted = detector.generate_synthetic_drift(
            n_samples=200,
            drift_type="mean_shift",
            drift_magnitude=2.0,
            features_to_drift=['feature1']
        )
        
        assert len(synthetic_drifted) == 200
        assert 'feature1' in synthetic_drifted.columns
        
        # Check that mean has shifted
        ref_mean = reference_data['feature1'].mean()
        synth_mean = synthetic_drifted['feature1'].mean()
        assert abs(synth_mean - ref_mean) > 1.0  # Should have significant shift
    
    def test_generate_synthetic_drift_variance_shift(self, reference_data):
        """Test synthetic drift generation with variance shift."""
        detector = DataDriftDetector(reference_data)
        
        synthetic_drifted = detector.generate_synthetic_drift(
            n_samples=200,
            drift_type="variance_shift",
            drift_magnitude=1.5,
            features_to_drift=['feature2']
        )
        
        # Check that variance has increased
        ref_std = reference_data['feature2'].std()
        synth_std = synthetic_drifted['feature2'].std()
        assert synth_std > ref_std * 1.2  # Should have increased variance
    
    def test_drift_score_calculation(self, reference_data):
        """Test drift score calculation."""
        detector = DataDriftDetector(reference_data)
        
        # No drift
        no_drift_data = reference_data.copy()
        results_no_drift = detector.detect_drift(no_drift_data)
        
        # High drift
        high_drift_data = reference_data.copy()
        high_drift_data['feature1'] = high_drift_data['feature1'] + 10
        high_drift_data['feature2'] = high_drift_data['feature2'] + 5
        results_high_drift = detector.detect_drift(high_drift_data)
        
        # Drift score should be higher for high drift
        assert results_high_drift['drift_score'] >= results_no_drift['drift_score']


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""
    
    @pytest.fixture
    def baseline_metrics(self):
        """Create baseline performance metrics."""
        return {
            'mae': 100.0,
            'rmse': 150.0,
            'r2': 0.85
        }
    
    def test_init(self, baseline_metrics):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(baseline_metrics, performance_threshold=0.2)
        
        assert monitor.baseline_metrics == baseline_metrics
        assert monitor.performance_threshold == 0.2
    
    def test_no_degradation(self, baseline_metrics):
        """Test when performance hasn't degraded."""
        monitor = PerformanceMonitor(baseline_metrics, metric_type="mae", performance_threshold=0.2)
        
        # Simulate predictions with similar performance
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = y_true + np.random.normal(0, 10, len(y_true))  # Small errors
        
        results = monitor.check_performance(y_true, y_pred)
        
        assert 'has_degradation' in results
        assert 'degradation_score' in results
        # Should not have significant degradation
        
    def test_mae_degradation(self, baseline_metrics):
        """Test detection of MAE degradation."""
        monitor = PerformanceMonitor(baseline_metrics, metric_type="mae", performance_threshold=0.2)
        
        # Simulate poor predictions (high MAE)
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = y_true + np.random.normal(0, 150, len(y_true))  # Large errors (MAE ~150)
        
        results = monitor.check_performance(y_true, y_pred)
        
        # Should detect degradation if MAE > 120 (20% worse than 100)
        if results['current_metrics']['mae'] > 120:
            assert results['has_degradation'] is True
            assert results['degradation_score'] > 0.2
    
    def test_r2_degradation(self, baseline_metrics):
        """Test detection of R² degradation."""
        monitor = PerformanceMonitor(baseline_metrics, metric_type="r2", performance_threshold=0.2)
        
        # Simulate poor predictions (low R²)
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.random.normal(300, 100, len(y_true))  # Random predictions (low R²)
        
        results = monitor.check_performance(y_true, y_pred)
        
        # Should detect degradation if R² < 0.68 (20% worse than 0.85)
        if results['current_metrics']['r2'] < 0.68:
            assert results['has_degradation'] is True
            assert results['degradation_score'] > 0.2
    
    def test_alert_threshold(self, baseline_metrics):
        """Test alert triggering for significant degradation."""
        monitor = PerformanceMonitor(baseline_metrics, metric_type="mae", performance_threshold=0.2)
        
        # Simulate very poor predictions
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = y_true + np.random.normal(0, 200, len(y_true))  # Very large errors
        
        results = monitor.check_performance(y_true, y_pred)
        
        # Alert should trigger if degradation > 1.5 * threshold (30%)
        if results['degradation_score'] > 0.3:
            assert results['alert'] is True
    
    def test_degradation_score_calculation(self, baseline_metrics):
        """Test degradation score calculation."""
        monitor = PerformanceMonitor(baseline_metrics, metric_type="mae", performance_threshold=0.2)
        
        # Perfect predictions (MAE = 0)
        y_true = np.array([100, 200, 300])
        y_pred = y_true.copy()
        
        results = monitor.check_performance(y_true, y_pred)
        
        # Degradation should be negative (better than baseline)
        assert results['degradation_score'] < 0
        assert results['has_degradation'] is False


class TestDataDriftWithSyntheticData:
    """Integration tests for drift detection with synthetic data."""
    
    def test_end_to_end_drift_detection(self):
        """Test complete drift detection workflow with synthetic data."""
        # Create reference data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(10, 2, 500),
            'feature2': np.random.normal(5, 1, 500),
            'target': np.random.normal(100, 20, 500)
        })
        
        # Initialize detector
        detector = DataDriftDetector(reference_data)
        
        # Generate synthetic drifted data
        synthetic_drifted = detector.generate_synthetic_drift(
            n_samples=200,
            drift_type="mean_shift",
            drift_magnitude=2.0,
            features_to_drift=['feature1', 'feature2']
        )
        
        # Detect drift
        results = detector.detect_drift(synthetic_drifted)
        
        # Should detect drift
        assert results['has_drift'] is True
        assert results['drift_score'] > 0.1
        assert results['summary']['drifted_features'] > 0
    
    def test_performance_degradation_with_drift(self):
        """Test performance monitoring with drifted data."""
        # Baseline metrics
        baseline_metrics = {'mae': 100.0, 'rmse': 150.0, 'r2': 0.85}
        monitor = PerformanceMonitor(baseline_metrics, metric_type="mae", performance_threshold=0.2)
        
        # Simulate predictions on drifted data (worse performance)
        y_true = np.array([100, 200, 300, 400, 500])
        # Predictions with systematic bias (simulating drift impact)
        y_pred = y_true * 0.7 + np.random.normal(0, 50, len(y_true))
        
        results = monitor.check_performance(y_true, y_pred)
        
        # Should detect performance degradation
        assert 'has_degradation' in results
        assert 'alert' in results
        
        # If degradation is significant, should alert
        if results['current_metrics']['mae'] > baseline_metrics['mae'] * 1.3:
            assert results['alert'] is True

