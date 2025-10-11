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

```
mlops-team-61-project/
├── config/                    # Configuration files
│   └── paths_config.py        # Path management
├── data/
│   ├── raw/                   # Original data (DVC tracked)
│   │   └── bike_sharing_modified.csv
│   ├── interim/               # Intermediate processed data
│   │   └── bike_sharing_clean.csv (generated)
│   └── processed/             # Final processed data for modeling
├── docs/                      # Documentation
│   ├── ML_Canvas.md           # ML Canvas (business requirements)
│   └── EDA_Summary.md         # EDA insights and findings
├── models/                    # Trained models
├── notebooks/                 # Jupyter notebooks
│   └── notebook.ipynb         # Main EDA notebook
├── reports/                   # Generated analysis
│   └── figures/               # Visualizations
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   ├── models/                # Model training scripts
│   ├── tools/                 # Utility functions
│   └── visualization/         # Visualization scripts
├── .dvc/                      # DVC configuration
├── requirements.txt           # Python dependencies
└── setup.py                   # Package setup
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Isaac1606/mlops-team-61-project.git
cd mlops-team-61-project
```

### 2. Create Virtual Environment (Python 3.12 recommended)
```bash
# Using conda (recommended)
conda create -n mlops-project python=3.12 -y
conda activate mlops-project

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Configure AWS Credentials (for DVC)
```bash
aws configure --profile MLOpsTeamMemberUser
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Region: us-east-1
# - Output format: json
```

### 5. Pull Data with DVC
```bash
dvc remote modify raw profile MLOpsTeamMemberUser
dvc pull -r raw
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

- [ ] **Production Scripts**
  - `src/data/make_dataset.py`
  - `src/data/preprocess.py`
  - `src/models/train_model.py`
  - `src/models/predict_model.py`
  - `src/models/evaluate_model.py`

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
- **Containerization:** Docker (planned)

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
conda create -n mlops-project python=3.12 -y
conda activate mlops-project

# 2. Install
cd mlops-team-61-project
pip install -e .

# 3. Configure AWS & DVC
aws configure --profile MLOpsTeamMemberUser
dvc remote modify raw profile MLOpsTeamMemberUser

# 4. Get data
dvc pull -r raw

# 5. Explore
jupyter notebook notebooks/notebook.ipynb
```

---

**Last Updated:** October 2025  
**Version:** 0.1 (Phase 1 Complete)
