# Copilot Instructions for MLOps Bike Sharing Project

## Project Overview

This is an MLOps project predicting hourly bike rental demand using the Capital Bikeshare dataset (2011-2012). The project follows a structured MLOps pipeline with data versioning, experiment tracking, and modular code organization.

## Architecture & Key Components

### Data Pipeline Architecture
- **DVC-tracked data**: Raw data stored in S3 bucket (`s3://mlops-team-61-bucket/data/raw`) with AWS profile `MLOpsTeamMemberUser`
- **Path management**: Use `config/paths_config.py` for all file paths - it defines `PROJECT_ROOT` and standardized path constants
- **Data flow**: `data/raw/` → `data/interim/` → `data/processed/` (interim and processed directories created at runtime)

### Project Structure Pattern
```
src/
├── data/          # Data processing modules (planned)
├── models/        # Model training/evaluation modules (planned) 
├── tools/         # Utility functions
└── visualization/ # Plotting and reporting modules
```

## Critical Developer Workflows

### Environment Setup
```bash
# Always use Python 3.12 - specified in README
conda create -n mlops-project python=3.12 -y
conda activate mlops-project
pip install -e .  # Installs as editable package
```

### Data Operations
```bash
# Configure AWS profile first (required for DVC)
aws configure --profile MLOpsTeamMemberUser
dvc remote modify raw profile MLOpsTeamMemberUser
dvc pull -r raw  # Pull data from S3
```

## Project-Specific Conventions

### Path Handling
- **Always import**: `from config.paths_config import PROJECT_ROOT, RAW_FILE_PATH`
- **Never hardcode paths** - use the centralized config for consistency across team
- **DVC integration**: Raw data is `.csv.dvc` tracked, actual CSV files are gitignored

### Target Variable Naming
- **Primary target**: `cnt` (total bike rentals)
- **Secondary targets**: `casual` and `registered` (user type breakdown)
- **Business metrics**: MAE < 400, RMSE < 600, MAPE < 15%, R² > 0.85

### Development Phase Awareness
- **Phase 1 Complete**: EDA, data cleaning, DVC setup
- **Phase 2 Current**: Feature engineering, preprocessing 
- **Future phases**: Model training (XGBoost primary), MLflow tracking, production scripts

### Notebook Conventions
- **Main EDA notebook**: `notebooks/notebook.ipynb` contains comprehensive exploratory analysis
- **Key findings documented**: Temporal patterns (hourly peaks 7-9am, 5-7pm), weather correlations, multicollinearity (temp/atemp = 0.99)

## Integration Points

### MLflow Integration (Planned)
- Experiment tracking for model comparison
- Model registry for versioning
- Metrics: MAE, RMSE, R², MAPE

### AWS S3 Integration
- **Bucket**: `mlops-team-61-bucket`
- **Profile**: `MLOpsTeamMemberUser` (team shared)
- **DVC remote**: Configured for data versioning

## Team Workflow & Development Practices

### Git & Collaboration
- **Main branch**: `main` - stable code only
- **Feature development**: Create feature branches for new work
- **Code coordination**: 4-person team (Data Engineer, Data Scientist, ML Engineer, DevOps)
- **Academic timeline**: Phase-based development (currently Phase 2 of 4)

### Data Processing Conventions
- **Naming patterns**: Use descriptive prefixes (`bike_sharing_clean.csv`, `bike_sharing_preprocessed.csv`)
- **Intermediate data**: Store in `data/interim/` with timestamp suffixes for versioning
- **Feature naming**: Use snake_case for engineered features (`temp_rolling_24h`, `hr_sin`, `hr_cos`)
- **Data validation**: Always check data shape and null counts after transformations

### Experiment Tracking (Current Approach)
- **Notebook-based**: Primary experimentation in `notebooks/notebook.ipynb`
- **Version control**: Use git commits to track experimental iterations
- **Documentation**: Record key findings in markdown cells within notebooks
- **Metrics tracking**: Document performance metrics in notebook outputs
- **Future**: MLflow integration planned for Phase 3

### Testing & Validation Patterns
- **Data validation**: Always verify data integrity after DVC pulls
- **Shape checks**: Validate expected dimensions (17,500+ rows, 16 features)
- **Null value monitoring**: Track null counts per column during preprocessing
- **Correlation validation**: Verify expected relationships (temp/atemp = 0.99)
- **Business logic**: Ensure `cnt = casual + registered` relationship holds

## Team Context
- **4-person team**: Data Engineer, Data Scientist, ML Engineer, DevOps
- **Academic project**: Tec de Monterrey MLOps course
- **Timeline**: Phase-based development (currently Phase 2)

## Important Notes

- **No CI/CD yet**: Project in early development phase, no automated testing/deployment
- **Package installation**: Always use `pip install -e .` for development (editable install)
- **Data versioning**: Raw data changes require DVC tracking and team coordination
- **Modular design**: Follow the planned `src/` structure even for early development
- **Reproducibility**: Always set random seeds for consistent results across team members