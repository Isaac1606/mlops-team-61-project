# Test Suite

This directory contains comprehensive tests for the Bike Sharing Demand Prediction project.

## Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Unit tests for individual components
│   ├── test_data_cleaner.py
│   ├── test_feature_engineering.py
│   ├── test_data_splitter.py
│   ├── test_model_evaluator.py
│   └── test_preprocessor.py
├── integration/             # Integration tests for end-to-end workflows
│   ├── test_pipeline.py
│   └── test_drift_monitoring.py
└── data_drift/              # Data drift detection tests
    └── test_data_drift.py
```

## Running Tests

### All Tests
```bash
make test
# or
pytest tests/ -v
```

### Unit Tests Only
```bash
make test-unit
# or
pytest tests/unit/ -v
```

### Integration Tests Only
```bash
make test-integration
# or
pytest tests/integration/ -v
```

### Data Drift Tests Only
```bash
make test-drift
# or
pytest tests/data_drift/ -v
```

### With Coverage Report
```bash
make test-coverage
# or
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### Fast Tests (Unit Only, Stop on First Failure)
```bash
make test-fast
# or
pytest tests/unit/ -v -x
```

## Test Coverage

- **Unit Tests**: Test individual components in isolation
  - Data cleaning operations
  - Feature engineering methods
  - Data splitting logic
  - Model evaluation metrics
  - Preprocessing transformers

- **Integration Tests**: Test complete workflows
  - End-to-end ML pipeline (data → features → train → evaluate)
  - Data drift detection with trained models
  - Performance monitoring on production data

- **Data Drift Tests**: Test drift detection and monitoring
  - Statistical drift detection (KS test, Chi-square, PSI)
  - Synthetic data generation for testing
  - Performance degradation detection
  - Combined drift and performance monitoring

## Key Features

### Data Drift Detection

The `DataDriftDetector` class provides:
- **Statistical Tests**: Kolmogorov-Smirnov (continuous), Chi-square (categorical)
- **PSI (Population Stability Index)**: Feature-level drift detection
- **Synthetic Data Generation**: Generate drifted data for testing
- **Comprehensive Reporting**: Detailed drift information per feature

### Performance Monitoring

The `PerformanceMonitor` class provides:
- **Baseline Comparison**: Compare current performance to baseline
- **Degradation Detection**: Alert when performance degrades
- **Multiple Metrics**: Support for MAE, RMSE, R²
- **Alert Thresholds**: Configurable degradation thresholds

## Example Usage

See `tests/integration/test_drift_monitoring.py` for examples of:
1. Detecting drift in production data
2. Monitoring performance degradation
3. Using synthetic data for testing
4. Combined drift and performance monitoring

## Dependencies

Tests require:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel execution (optional)
- `scipy` - Statistical tests for drift detection

Install with:
```bash
pip install pytest pytest-cov pytest-xdist scipy
```

