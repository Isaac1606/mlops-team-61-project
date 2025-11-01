# 🏗️ Project Architecture

This document describes the architecture of the Bike Sharing Demand Prediction project, following MLOps best practices.

## 📐 Project Structure

```
mlops-team-61-project/
├── config/                      # Configuration files
│   ├── config.yaml              # Central configuration (YAML)
│   └── paths_config.py          # Legacy path config (deprecated)
│
├── data/                        # Data directory (DVC tracked)
│   ├── raw/                     # Raw data files
│   ├── interim/                 # Intermediate processed data
│   └── processed/               # Final processed datasets
│
├── models/                      # Trained models (DVC tracked)
│   ├── *.pkl                    # Saved models
│   └── *_feature_importance.csv # Feature importance reports
│
├── src/                         # Source code (Python package)
│   ├── config/                  # Configuration management
│   │   ├── config_loader.py     # YAML config loader
│   │   └── paths.py             # Path management
│   │
│   ├── data/                    # Data processing module
│   │   ├── data_loader.py       # Data loading utilities
│   │   ├── data_cleaner.py     # Data cleaning operations
│   │   ├── feature_engineering.py # Feature engineering
│   │   ├── data_splitter.py    # Temporal data splitting
│   │   └── make_dataset.py     # Data processing pipeline script
│   │
│   ├── models/                  # Modeling module
│   │   ├── preprocessor.py     # Scikit-Learn preprocessor
│   │   ├── pipeline.py          # Scikit-Learn pipeline wrapper
│   │   ├── model_trainer.py    # Model training with MLflow
│   │   ├── model_evaluator.py  # Model evaluation utilities
│   │   └── train_model.py      # Training pipeline script
│   │
│   └── tools/                   # Utility functions (placeholder)
│
├── notebooks/                   # Jupyter notebooks (exploratory)
│   ├── notebook.ipynb           # EDA notebook
│   └── 02_modeling.ipynb       # Modeling notebook
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # This file
│   ├── ML_Canvas.md            # Business requirements
│   └── EDA_Summary.md          # EDA findings
│
├── reports/                     # Generated reports
│   └── figures/                # Visualizations
│
├── mlruns/                      # MLflow tracking data (gitignored)
│
├── config.yaml                  # Main configuration file
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment file
├── setup.py                    # Package setup script
├── Makefile                    # Make commands for reproducibility
├── .dvc/                       # DVC configuration
└── README.md                   # Project README
```

## 🔄 Data Flow

```
Raw Data (CSV)
    ↓
[DataLoader] → Load raw data
    ↓
[DataCleaner] → Clean data (types, nulls, outliers)
    ↓
[FeatureEngineer] → Create features (lags, rolling, cyclical, interactions)
    ↓
[DataSplitter] → Split temporally (train/val/test)
    ↓
[DataPreprocessor] → Scale features (RobustScaler)
    ↓
[MLPipeline] → Preprocess + Model (Scikit-Learn Pipeline)
    ↓
[ModelTrainer] → Train with MLflow tracking
    ↓
[ModelEvaluator] → Evaluate metrics
    ↓
Trained Models + MLflow Experiments
```

## 🎯 Design Principles

### 1. **Single Responsibility Principle**
Each class has one clear purpose:
- `DataLoader`: Loading data from files
- `DataCleaner`: Cleaning operations
- `FeatureEngineer`: Feature creation
- `ModelTrainer`: Training logic
- `ModelEvaluator`: Evaluation metrics

### 2. **Dependency Injection**
Classes receive dependencies via constructor:
```python
config = ConfigLoader()
paths = ProjectPaths(config)
trainer = ModelTrainer(config, paths)
```

### 3. **Scikit-Learn Compatibility**
All transformers implement `fit()`/`transform()` pattern:
```python
preprocessor = DataPreprocessor()
preprocessor.fit(X_train)
X_train_scaled = preprocessor.transform(X_train)
```

### 4. **Configuration-Driven**
All parameters come from `config/config.yaml`:
- No hardcoded values
- Easy to experiment
- Version controlled

### 5. **Reproducibility**
- Fixed random seeds
- DVC for data versioning
- MLflow for experiment tracking
- Environment files for dependencies

## 🔧 Core Components

### Configuration Management

**`ConfigLoader`** (`src/config/config_loader.py`)
- Loads YAML configuration
- Provides typed access to config values
- Centralizes all parameters

**`ProjectPaths`** (`src/config/paths.py`)
- Manages all file paths
- Creates directories as needed
- Follows Cookiecutter structure

### Data Processing

**`DataLoader`** (`src/data/data_loader.py`)
- Loads raw and processed data
- Validates file existence
- Handles errors gracefully

**`DataCleaner`** (`src/data/data_cleaner.py`)
- Converts data types
- Handles null values
- Removes problematic columns

**`FeatureEngineer`** (`src/data/feature_engineering.py`)
- Creates lag features
- Rolling statistics
- Cyclical encodings
- Interaction features
- Advanced features (volatility, momentum)

**`DataSplitter`** (`src/data/data_splitter.py`)
- Temporal splitting (respects time order)
- Configurable split ratios
- Prevents data leakage

### Modeling

**`DataPreprocessor`** (`src/models/preprocessor.py`)
- Scikit-Learn compatible transformer
- Handles scaling (StandardScaler, RobustScaler, MinMaxScaler)
- Excludes binary/categorical features from scaling

**`MLPipeline`** (`src/models/pipeline.py`)
- Wraps preprocessing + model in Scikit-Learn Pipeline
- Ensures consistent transformations
- Prevents data leakage

**`ModelTrainer`** (`src/models/model_trainer.py`)
- Creates models from config
- Trains with MLflow tracking
- Saves trained models
- Logs hyperparameters and metrics

**`ModelEvaluator`** (`src/models/model_evaluator.py`)
- Computes multiple metrics (MAE, RMSE, R², MAPE)
- Compares against targets
- Generates evaluation reports
- Extracts feature importance

## 🔬 Scikit-Learn Pipeline Integration

The project uses **Scikit-Learn Pipelines** for end-to-end ML workflows:

```python
from src.models import MLPipeline
from sklearn.ensemble import RandomForestRegressor

# Create pipeline
pipeline = MLPipeline(
    model=RandomForestRegressor(n_estimators=100),
    preprocessor_config={
        "scaler_type": "robust",
        "exclude_from_scaling": ["holiday", "workingday"]
    }
)

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
```

**Benefits:**
- ✅ Prevents data leakage (transformations learned only on train)
- ✅ Single object to serialize for deployment
- ✅ Consistent transformations across train/test
- ✅ Easy to integrate into production systems

## 📊 MLflow Integration

MLflow is used for:
1. **Experiment Tracking**: All hyperparameters and metrics
2. **Model Registry**: Versioned model storage
3. **Reproducibility**: Full experiment context saved
4. **Comparison**: Compare different model runs

**Example Usage:**
```python
from src.models import ModelTrainer

trainer = ModelTrainer(config, paths)
pipeline = trainer.train_model(
    model_type="xgboost",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    run_name="xgboost_baseline"
)
```

All hyperparameters, metrics, and models are automatically logged to MLflow.

## 🔄 DVC Integration

DVC is used for:
1. **Data Versioning**: Track raw and processed datasets
2. **Model Versioning**: Track trained models
3. **Remote Storage**: S3 integration for large files

**Commands:**
```bash
# Pull data from remote
dvc pull -r raw

# Track new data
dvc add data/raw/bike_sharing_modified.csv

# Push to remote
dvc push -r raw
```

## 🚀 Production-Ready Scripts

The project includes executable scripts:

1. **`src/data/make_dataset.py`**
   - Complete data processing pipeline
   - Can be run standalone: `python src/data/make_dataset.py`
   - Or via Makefile: `make data`

2. **`src/models/train_model.py`**
   - Complete training pipeline
   - Trains multiple models
   - Logs to MLflow
   - Can be run standalone or via Makefile

## 📦 Package Structure

The project is structured as a **Python package**:

```python
# Install in editable mode
pip install -e .

# Import modules
from src.config import ConfigLoader, ProjectPaths
from src.data import DataLoader, DataCleaner
from src.models import ModelTrainer, MLPipeline
```

## 🔒 Reproducibility Features

1. **Fixed Random Seeds**: Configured in `config.yaml`
2. **Environment Files**: `environment.yml` and `requirements.txt`
3. **Version Control**: Git for code, DVC for data
4. **Configuration**: All parameters in `config.yaml`
5. **Experiment Tracking**: MLflow tracks every experiment
6. **Makefile**: Standardized commands via `make`

## 🎓 Best Practices Implemented

- ✅ **Cookiecutter Structure**: Standardized project layout
- ✅ **OOP Design**: Classes with single responsibility
- ✅ **Scikit-Learn Pipelines**: Production-ready ML workflows
- ✅ **MLflow Integration**: Professional experiment tracking
- ✅ **DVC Integration**: Data and model versioning
- ✅ **Configuration Management**: Centralized parameters
- ✅ **Type Hints**: Better code documentation
- ✅ **Logging**: Comprehensive logging throughout
- ✅ **Error Handling**: Graceful error handling
- ✅ **Documentation**: Comprehensive docstrings

## 🔮 Future Enhancements

Potential additions:
- [ ] Unit tests (pytest)
- [ ] Integration tests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Docker containerization
- [ ] API deployment (FastAPI)
- [ ] Model monitoring (Evidently AI)
- [ ] Automated hyperparameter tuning (Optuna)

---

**Last Updated:** 2025-01-13  
**Version:** 0.1.0

