# Model package initialization
# Makes the models directory a Python package

from .base_model import BaseModel, setup_mlflow
from .linear_regression_model import LinearRegressionModel
from .random_forest_model import RandomForestModel

try:
    from .xgboost_model import XGBoostModel
    __all__ = ['BaseModel', 'setup_mlflow', 'LinearRegressionModel', 'RandomForestModel', 'XGBoostModel']
except ImportError:
    __all__ = ['BaseModel', 'setup_mlflow', 'LinearRegressionModel', 'RandomForestModel']

__version__ = '0.1.0'