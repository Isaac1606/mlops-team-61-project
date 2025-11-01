# 🔧 Troubleshooting Guide

Common issues and solutions for the Bike Sharing Demand Prediction project.

## XGBoost Installation Issues

### Error: "XGBoost Library (libxgboost.dylib) could not be loaded"

**Symptoms:**
```
XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed
    - libomp.dylib for Mac OSX
Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
```

**Solution (macOS):**

1. Install OpenMP runtime using Homebrew:
   ```bash
   brew install libomp
   ```

2. If Homebrew is not installed, install it first:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. Reinstall XGBoost to ensure it finds the library:
   ```bash
   pip install --upgrade --force-reinstall xgboost
   ```

**Alternative Solutions:**

1. **Use conda instead of pip:**
   ```bash
   conda install -c conda-forge xgboost
   ```
   Conda handles OpenMP dependencies automatically.

2. **Continue without XGBoost:**
   The code is designed to work without XGBoost. The training script will automatically skip XGBoost if it's not available and train only Ridge and Random Forest models.

**Verification:**
```python
python -c "import xgboost; print('XGBoost installed successfully')"
```

---

## Module Import Errors

### Error: "ModuleNotFoundError: No module named 'src'"

**Solution:**
Install the package in editable mode:
```bash
pip install -e .
```

Or ensure the project root is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Data File Not Found Errors

### Error: "FileNotFoundError: Raw data file not found"

**Solution:**
Pull data from DVC remote:
```bash
make dvc-pull
# or
dvc pull -r raw --force
```

If you don't have DVC configured, check:
1. AWS credentials are set up: `aws configure --profile MLOpsTeamMemberUser`
2. DVC remote is configured: `dvc remote list`

---

## MLflow UI Not Showing Experiments

### Issue: MLflow UI shows no experiments

**Possible Causes:**

1. **Wrong tracking URI:**
   - Check `config/config.yaml` → `mlflow.tracking_uri`
   - Ensure path is correct (relative or absolute)

2. **Experiments not logged:**
   - Verify training script ran successfully
   - Check `mlruns/` directory exists and contains data

3. **Port conflict:**
   ```bash
   # Use different port
   mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5001
   ```

**Verification:**
```bash
# Check if mlruns directory exists and has content
ls -la mlruns/

# Start MLflow UI and check logs
make mlflow-ui
```

---

## Configuration Errors

### Error: "Configuration file not found: config/config.yaml"

**Solution:**
Ensure you're running commands from the project root directory:
```bash
cd /path/to/mlops-team-61-project
python src/models/train_model.py
```

---

## Dependency Conflicts

### Error: Package version conflicts

**Solution:**
1. Use conda environment (recommended):
   ```bash
   conda env create -f environment.yml
   conda activate mlops-team-61
   ```

2. Or create fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   pip install -e .
   ```

---

## Python Version Issues

### Error: "Python version not supported"

**Requirements:**
- Python 3.9+ (Python 3.12 recommended)

**Check version:**
```bash
python --version
```

**Update if needed:**
```bash
# Using pyenv (recommended)
pyenv install 3.12.2
pyenv local 3.12.2

# Or using conda
conda install python=3.12
```

---

## DVC Remote Access Issues

### Error: "Access Denied" or "Credentials not found"

**Solution:**

1. **Configure AWS credentials:**
   ```bash
   aws configure --profile MLOpsTeamMemberUser
   ```

2. **Verify DVC remote:**
   ```bash
   dvc remote list
   dvc remote modify raw profile MLOpsTeamMemberUser
   ```

3. **Test connection:**
   ```bash
   dvc pull -r raw --force
   ```

---

## Memory Issues

### Error: "MemoryError" or "Killed" during training

**Solutions:**

1. **Reduce model complexity:**
   - Edit `config/config.yaml`
   - Reduce `n_estimators`, `max_depth` in model configs

2. **Use fewer features:**
   - Modify feature engineering config
   - Remove expensive features (rolling windows, etc.)

3. **Train models individually:**
   ```bash
   # Instead of make train, run individual models
   python -c "from src.models.train_model import main; main()"
   # Then modify script to train one model at a time
   ```

---

## Permission Errors

### Error: "Permission denied" when writing files

**Solution:**
Check file permissions:
```bash
# Make sure you have write access
ls -la models/
chmod -R u+w models/ data/processed/
```

---

## Random Seed Issues

### Results differ from expected

**Check:**
1. Random seed in `config/config.yaml`:
   ```yaml
   reproducibility:
     seed: 42
   ```

2. Ensure seed is set before training:
   ```python
   import numpy as np
   import random
   
   np.random.seed(42)
   random.seed(42)
   ```

3. Check package versions (may affect random number generation):
   ```bash
   pip list | grep -E "numpy|scikit-learn|pandas"
   ```

---

## Quick Diagnostic Commands

```bash
# Check Python version
python --version

# Check package installation
pip list | grep -E "xgboost|mlflow|dvc|pandas|sklearn"

# Check XGBoost specifically
python -c "import xgboost; print(xgboost.__version__)"

# Check project structure
ls -la src/
ls -la config/

# Check data availability
ls -la data/raw/
ls -la data/processed/

# Test imports
python -c "from src.config import ConfigLoader; print('Config OK')"
python -c "from src.data import DataLoader; print('Data OK')"
python -c "from src.models import ModelTrainer; print('Models OK')"
```

---

## Getting Help

If you encounter an issue not covered here:

1. **Check logs:** Look at the error message carefully
2. **Check configuration:** Verify `config/config.yaml` settings
3. **Check dependencies:** Ensure all packages are installed correctly
4. **Check environment:** Verify Python version and environment setup
5. **Review documentation:** See `REPRODUCIBILITY.md` and `docs/ARCHITECTURE.md`

---

**Last Updated:** 2025-01-13

