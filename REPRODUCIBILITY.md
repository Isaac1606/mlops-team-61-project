# 🔄 Reproducibility Guide

This document ensures that **anyone** can reproduce the experiments and obtain equivalent results.

## ✅ Prerequisites

1. **Python 3.9+** (Python 3.12 recommended)
2. **Git** (for version control)
3. **DVC** (for data versioning)
4. **AWS CLI** (for S3 access, if using remote storage)

## 🚀 Quick Start (Reproducible Setup)

### Option 1: Using Conda (Recommended for Reproducibility)

```bash
# 1. Clone the repository
git clone https://github.com/Isaac1606/mlops-team-61-project.git
cd mlops-team-61-project

# 2. Create conda environment from environment.yml
conda env create -f environment.yml
conda activate mlops-team-61

# 3. Install package in editable mode
pip install -e .

# 4. Pull data from DVC
make dvc-pull

# 5. Run complete pipeline
make all
```

### Option 2: Using pip + virtualenv

```bash
# 1. Clone the repository
git clone https://github.com/Isaac1606/mlops-team-61-project.git
cd mlops-team-61-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
make install

# 4. Pull data from DVC
make dvc-pull

# 5. Run complete pipeline
make all
```

## 📋 Step-by-Step Pipeline

### 1. Data Processing

```bash
# Process raw data → clean → features → train/val/test splits
python src/data/make_dataset.py

# Or using Makefile
make data
```

**Expected outputs:**
- `data/interim/bike_sharing_clean.csv` - Cleaned data
- `data/processed/bike_sharing_features_train.csv` - Training set
- `data/processed/bike_sharing_features_val.csv` - Validation set
- `data/processed/bike_sharing_features_test.csv` - Test set

### 2. Model Training

```bash
# Train models (Ridge, Random Forest, XGBoost) with MLflow tracking
python src/models/train_model.py

# Or using Makefile
make train
```

**Expected outputs:**
- `models/ridge_baseline.pkl` - Trained Ridge model
- `models/random_forest_baseline.pkl` - Trained RF model
- `models/xgboost_baseline.pkl` - Trained XGBoost model
- `models/*_feature_importance.csv` - Feature importance files
- `mlruns/` - MLflow experiment tracking data

### 3. View Results

```bash
# Start MLflow UI
make mlflow-ui

# Open browser to http://localhost:5000
```

## 🔧 Configuration

All configuration is centralized in `config/config.yaml`. This ensures:
- ✅ **Reproducibility**: Same config = same results
- ✅ **Version Control**: Config changes are tracked in git
- ✅ **Flexibility**: Easy to experiment with different settings

### Key Configuration Sections

- **`data`**: Data paths, split ratios, target column
- **`features`**: Feature engineering parameters (lags, rolling windows, etc.)
- **`models`**: Model hyperparameters (Ridge, RF, XGBoost)
- **`mlflow`**: Experiment tracking settings
- **`evaluation`**: Target metrics and thresholds

## 🔬 Reproducing Specific Experiments

### Reproduce with Exact Same Random Seed

The random seed is configured in `config/config.yaml`:
```yaml
reproducibility:
  seed: 42
  n_jobs: -1
```

All models use this seed, ensuring reproducible results.

### Reproduce Specific Model

```python
from src.config import ConfigLoader, ProjectPaths
from src.models import ModelTrainer, ModelEvaluator
from src.data import DataLoader

# Load config
config = ConfigLoader()
paths = ProjectPaths(config)

# Load data
data_loader = DataLoader(paths)
train_df = data_loader.load_processed_data("train")
val_df = data_loader.load_processed_data("val")

# Train specific model
trainer = ModelTrainer(config, paths)
pipeline = trainer.train_model(
    model_type="xgboost",  # or "ridge", "random_forest"
    X_train=train_df.drop(columns=["cnt"]),
    y_train=train_df["cnt"],
    X_val=val_df.drop(columns=["cnt"]),
    y_val=val_df["cnt"]
)
```

## 📊 Expected Results

With the current configuration, you should expect:

| Model | Validation MAE | Validation RMSE | Validation R² |
|-------|----------------|-----------------|----------------|
| Ridge | ~90-100 | ~160-170 | ~0.55-0.60 |
| Random Forest | ~40-50 | ~100-110 | ~0.82-0.85 |
| XGBoost | ~35-40 | ~70-75 | ~0.92-0.93 |

**Note:** Results may vary slightly due to:
- Operating system differences
- NumPy/Scikit-Learn version differences
- Hardware differences (floating point precision)

However, results should be **within 1-2%** of the expected values.

## 🔍 Troubleshooting

### Issue: XGBoost Library Not Found

**Symptoms:**
```
XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
```

**Solution (macOS):**
```bash
# Install OpenMP runtime
brew install libomp

# Reinstall XGBoost
pip install --upgrade --force-reinstall xgboost
```

**Note:** The code will automatically skip XGBoost if it's not available and train other models.

**See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more details.**

### Issue: "FileNotFoundError: Raw data file not found"

**Solution:** Pull data from DVC remote:
```bash
make dvc-pull
```

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:** Install package in editable mode:
```bash
pip install -e .
```

### Issue: MLflow UI not showing experiments

**Solution:** Check that MLflow tracking URI is correct:
```python
# In config/config.yaml
mlflow:
  tracking_uri: "file:///mlruns"  # Relative to project root
```

### Issue: Results differ from expected

**Solution:** 
1. Check that you're using the same config (`config/config.yaml`)
2. Verify random seed is set correctly
3. Check versions of key packages:
   ```bash
   pip list | grep -E "scikit-learn|numpy|pandas|xgboost"
   ```

## 🧪 Verification Checklist

Before considering an experiment reproducible, verify:

- [ ] Same Python version (check with `python --version`)
- [ ] Same package versions (check with `pip freeze > requirements_lock.txt`)
- [ ] Same random seed (check `config/config.yaml`)
- [ ] Same data (check DVC hash: `dvc status`)
- [ ] Same configuration (check `config/config.yaml` git commit)
- [ ] Results match expected values (within 1-2%)

## 📝 Environment Locking

For **maximum reproducibility**, create a locked requirements file:

```bash
pip freeze > requirements_lock.txt
```

Then install from locked file:
```bash
pip install -r requirements_lock.txt
```

## 🔗 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

---

**Last Updated:** 2025-01-13  
**Version:** 0.1.0

