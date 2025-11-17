# 🚴 Bike Sharing Demand Prediction - MLOps Project

**Team 61** | Tec de Monterrey | MLOps Master's Program

---

## 📊 Project Overview

Machine Learning system for predicting hourly bike rental demand in the Capital Bikeshare system (Washington D.C., 2011-2012). This project implements end-to-end MLOps practices including data versioning, experiment tracking, and model deployment strategies.

### Business Objective
- Predict bike demand 1-24 hours ahead for operational optimization
- Enable dynamic pricing and resource allocation
- Reduce operational costs by 20% and increase revenue by 12%

### Target Metrics
- **MAE** < 400 bikes/hour (target: < 300)
- **RMSE** < 600
- **MAPE** < 15%
- **R²** > 0.85

---

## 🗂️ Project Structure

This project follows **Cookiecutter Data Science** best practices with a modern MLOps architecture:

```
mlops-team-61-project/
├── config/                     # Configuration files
│   ├── config.yaml            # Central configuration (YAML)
│   └── paths_config.py        # Legacy path config
├── data/                       # Data directory (DVC tracked)
│   ├── raw/                   # Original data (DVC tracked)
│   │   └── bike_sharing_modified.csv
│   ├── interim/               # Intermediate processed data
│   │   └── bike_sharing_clean.csv
│   └── processed/             # Final processed data for modeling
│       ├── bike_sharing_features_train.csv
│       ├── bike_sharing_features_val.csv
│       └── bike_sharing_features_test.csv
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # Project architecture (NEW)
│   ├── ML_Canvas.md           # ML Canvas (business requirements)
│   └── EDA_Summary.md         # EDA insights and findings
├── models/                     # Trained models (DVC tracked)
│   ├── *.pkl                  # Saved models
│   └── *_feature_importance.csv
├── notebooks/                  # Jupyter notebooks (exploratory)
│   ├── notebook.ipynb         # Main EDA notebook
│   └── 02_modeling.ipynb      # Modeling notebook
├── reports/                    # Generated analysis
│   └── figures/               # Visualizations
├── src/                        # Source code (Python package)
│   ├── config/                # Configuration management
│   │   ├── config_loader.py   # YAML config loader
│   │   └── paths.py          # Path management
│   ├── data/                  # Data processing module
│   │   ├── data_loader.py     # Data loading utilities
│   │   ├── data_cleaner.py    # Data cleaning operations
│   │   ├── feature_engineering.py # Feature engineering
│   │   ├── data_splitter.py  # Temporal data splitting
│   │   └── make_dataset.py   # Data processing pipeline script
│   ├── models/                # Modeling module
│   │   ├── preprocessor.py   # Scikit-Learn preprocessor
│   │   ├── pipeline.py       # Scikit-Learn pipeline wrapper
│   │   ├── model_trainer.py   # Model training with MLflow
│   │   ├── model_evaluator.py # Model evaluation utilities
│   │   └── train_model.py    # Training pipeline script
│   └── tools/                 # Utility functions
├── .dvc/                       # DVC configuration
├── mlruns/                     # MLflow tracking data (gitignored)
├── config.yaml                 # Main configuration file (NEW)
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment file (NEW)
├── setup.py                    # Package setup
├── Makefile                    # Make commands for reproducibility (NEW)
└── REPRODUCIBILITY.md          # Reproducibility guide (NEW)
```

**Key Improvements:**
- ✅ **Cookiecutter Structure**: Standardized project layout
- ✅ **OOP Design**: Object-oriented classes for modularity
- ✅ **Scikit-Learn Pipelines**: Production-ready ML workflows
- ✅ **Configuration Management**: Centralized YAML config
- ✅ **Reproducibility**: Makefile, environment files, documentation

---

## ⚙️ Setup Instructions

### Quick Start (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/Isaac1606/mlops-team-61-project.git
cd mlops-team-61-project

# 2. Create conda environment (recommended for reproducibility)
conda env create -f environment.yml
conda activate mlops-team-61

# 3. Install package
pip install -e .

# 4. Pull data from DVC
make dvc-pull

# 5. Run complete pipeline
make all
```

### Alternative Setup (pip + venv)

```bash
# 1. Clone repository
git clone https://github.com/Isaac1606/mlops-team-61-project.git
cd mlops-team-61-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate   # Windows

# 3. Install dependencies
make install

# 4. Pull data from DVC
make dvc-pull

# 5. Run complete pipeline
make all
```

### Configure AWS Credentials (for DVC)

```bash
aws configure --profile MLOpsTeamMemberUser
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Region: us-east-1
# - Output format: json
```

---

## 📈 Current Progress

### ✅ Completed (Phase 1)

- [x] **Project Setup**
  - Environment configuration with Python 3.12
  - All dependencies installed
  - AWS S3 integration for data versioning

- [x] **ML Canvas**
  - Business requirements documented
  - Value proposition defined
  - Metrics and KPIs established
  - Technical architecture designed

- [x] **Data Versioning**
  - DVC configured with S3 remote
  - Dataset tracked and versioned
  - ~1.6 MB dataset (17,726 observations)

- [x] **Data Quality & Cleaning**
  - All columns converted to correct data types
  - 180-237 null values per column handled
  - Problematic columns removed (instant, mixed_type_col)
  - Final dataset: ~17,500 clean observations

- [x] **Exploratory Data Analysis (EDA)**
  - Comprehensive univariate and multivariate analysis
  - Temporal patterns identified (hourly, daily, seasonal)
  - Weather impact quantified
  - User type behavior analyzed (casual vs registered)
  - Feature correlations calculated
  - Outliers detected and documented

**Key Findings:**
- 📊 Strong hourly patterns with peaks at 7-9am and 5-7pm
- 🌤️ Temperature correlation: +0.40 (strong positive)
- 📈 40% demand growth from 2011 to 2012
- 👥 80% registered users, 20% casual users
- ⚠️ Multicollinearity detected: temp and atemp (0.99)

---

### 🔄 In Progress (Phase 2)

- [ ] **Feature Engineering**
  - Cyclical features (sin/cos for hour, month)
  - Lag features (t-1, t-24)
  - Rolling averages (7-day, 24-hour)
  - Interaction features (temp×season, hr×workingday)
  - One-hot encoding for categoricals

- [ ] **Data Preprocessing**
  - Temporal train/validation/test split
  - Feature scaling (StandardScaler)
  - Save to `data/processed/`
  - Version with DVC

---

### 📝 Pending (Phase 3 & 4)

- [ ] **Modeling**
  - Baseline: Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor (primary model)
  - Hyperparameter tuning
  - MLflow experiment tracking

- [ ] **Evaluation & Validation**
  - Performance metrics (MAE, RMSE, R², MAPE)
  - Temporal cross-validation
  - Feature importance analysis
  - Prediction vs actual visualizations

- [x] **Production Scripts**
  - [x] `src/data/make_dataset.py` - Complete data processing pipeline
  - [x] `src/models/train_model.py` - Complete training pipeline with MLflow
  - [ ] `src/models/predict_model.py` - Inference script (planned)

- [ ] **Documentation & Presentation**
  - Executive presentation (PDF)
  - Process documentation
  - Team roles and responsibilities
  - Final report

---

## 📊 Dataset Information

**Source:** Capital Bikeshare (Washington D.C.)  
**Period:** 2011-2012  
**Granularity:** Hourly  
**Total Records:** ~17,500 (after cleaning)

### Features (16 total)

**Temporal Features:**
- `dteday`: Date
- `season`: Season (1: winter, 2: spring, 3: summer, 4: fall)
- `yr`: Year (0: 2011, 1: 2012)
- `mnth`: Month (1-12)
- `hr`: Hour (0-23)
- `weekday`: Day of week (0-6)
- `holiday`: Is holiday (0/1)
- `workingday`: Is working day (0/1)

**Weather Features:**
- `weathersit`: Weather situation (1-4, clear to heavy rain)
- `temp`: Normalized temperature
- `atemp`: Normalized feeling temperature
- `hum`: Normalized humidity
- `windspeed`: Normalized wind speed

**Target Variables:**
- `cnt`: **Total bike rentals** (PRIMARY TARGET)
- `casual`: Casual user rentals
- `registered`: Registered user rentals

---

## 🛠️ Technology Stack

- **Language:** Python 3.12
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML Libraries:** Scikit-learn, XGBoost
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **Cloud Storage:** AWS S3
- **Orchestration:** Apache Airflow (planned)
- **Containerization:** Docker ✅

---

## 👥 Team & Roles

**Roles:**
- Data Engineer: Data pipeline, DVC setup /
- Data Scientist: EDA, modeling, evaluation
- ML Engineer: MLflow, deployment, monitoring / Gairo Peralta (gairo@berkeley.edu)
- DevOps: CI/CD, containerization / Isaac Carballo (isaac-dx@live.com.mx)

---

## 📚 Documentation

- [`docs/ML_Canvas.md`](docs/ML_Canvas.md) - Business requirements and ML design
- [`docs/EDA_Summary.md`](docs/EDA_Summary.md) - Exploratory analysis findings
- [`docs/DATA_DRIFT.md`](docs/DATA_DRIFT.md) - Data drift detection and performance monitoring guide ⭐ **NEW**
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Project architecture and design decisions
- [`notebooks/notebook.ipynb`](notebooks/notebook.ipynb) - Interactive EDA notebook

---

## 🔗 Resources

- [Original Dataset Information](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Project Repository](https://github.com/Isaac1606/mlops-team-61-project)

---

## 📜 License

This project is part of the MLOps course at Tec de Monterrey.

---

## 🚀 Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate mlops-team-61

# 2. Install package
pip install -e .

# 3. Configure AWS & DVC
aws configure --profile MLOpsTeamMemberUser
dvc remote modify raw profile MLOpsTeamMemberUser

# 4. Pull data
make dvc-pull

# 5. Run complete pipeline
make all

# 6. View results in MLflow
make mlflow-ui
# Open http://localhost:5000
```

## 🐳 Docker Deployment

The ML service can be containerized and deployed using Docker. This provides a reproducible, isolated environment for production deployments.

### Quick Start with Docker

```bash
# Build the Docker image
docker build -t ml-service:latest .

# Run the container
docker run -p 8000:8000 ml-service:latest
```

The service will be available at: `http://localhost:8000`

### Using Docker Compose (Recommended)

```bash
# Start ML service + Redis
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f ml-service

# Stop services
docker-compose down
```

### Publishing to Docker Hub

```bash
# 1. Login to Docker Hub
docker login

# 2. Tag the image
docker tag ml-service:latest YOUR_USERNAME/ml-service:latest
docker tag ml-service:latest YOUR_USERNAME/ml-service:v1.0.0

# 3. Push to Docker Hub
docker push YOUR_USERNAME/ml-service:latest
docker push YOUR_USERNAME/ml-service:v1.0.0
```

### Versioning Strategy

We use semantic versioning for container tags:
- `latest`: Most recent stable version
- `v1.0.0`: Specific version (major.minor.patch)
- `v1.0.0-beta`: Pre-release versions

**Example:**
```bash
docker build -t ml-service:v1.0.0 -t ml-service:latest .
docker tag ml-service:v1.0.0 YOUR_USERNAME/ml-service:v1.0.0
docker tag ml-service:v1.0.0 YOUR_USERNAME/ml-service:latest
docker push YOUR_USERNAME/ml-service:v1.0.0
docker push YOUR_USERNAME/ml-service:latest
```

For detailed Docker documentation, see **[DOCKER.md](DOCKER.md)**.

## 📖 Additional Documentation

- **[DOCKER.md](DOCKER.md)** - Complete Docker guide (build, run, publish) ⭐ **NEW**
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed architecture documentation
- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** - Reproducibility guide
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Troubleshooting common issues
- **[docs/ML_Canvas.md](docs/ML_Canvas.md)** - Business requirements
- **[docs/EDA_Summary.md](docs/EDA_Summary.md)** - EDA findings

## 🔄 Available Commands (Makefile)

```bash
make help          # Show all available commands
make install       # Install dependencies
make data          # Run data processing pipeline
make train         # Train models
make all           # Run complete pipeline (data + train)
make mlflow-ui     # Start MLflow UI
make dvc-pull      # Pull data from DVC
make dvc-push      # Push data to DVC
make clean         # Clean generated files

# Testing
make test          # Run all tests
make test-unit     # Run unit tests only
make test-integration  # Run integration tests
make test-drift    # Run data drift tests ⭐
make test-coverage # Run tests with coverage report
make test-fast     # Run fast tests (unit only, stop on first failure)
```

## 🧪 Testing

El proyecto incluye una suite completa de tests:

- **Unit Tests**: Tests para componentes individuales (34 tests)
- **Integration Tests**: Tests end-to-end del pipeline completo
- **Data Drift Tests**: Tests para detección de drift y monitoreo de performance (16 tests)

### Ejecutar Tests de Data Drift

```bash
# Todos los tests de drift
make test-drift

# Tests específicos
pytest tests/data_drift/test_data_drift.py::TestDataDriftDetector::test_mean_shift_drift -v
pytest tests/data_drift/ -v

# Ver documentación completa
cat docs/DATA_DRIFT.md
```

Para más información sobre data drift, consulta la [documentación completa](docs/DATA_DRIFT.md).

---

**Last Updated:** October 2025  
**Version:** 0.1 (Phase 1 Complete)
