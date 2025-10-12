# Machine Learning Models for Bike Sharing Demand Prediction
# Base classes and utilities for model training and evaluation

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Any
import mlflow
import mlflow.sklearn


class BaseModel(ABC):
    """
    Base abstract class for all models in the bike sharing prediction project.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def create_model(self, **params) -> Any:
        """Create and return the model instance"""
        pass
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Train the model"""
        pass
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
        
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def evaluate_objectives(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Evaluate if business objectives are met"""
        return {
            'mae_ok': metrics['mae'] < 400,
            'rmse_ok': metrics['rmse'] < 600,
            'r2_ok': metrics['r2'] > 0.85,
            'mape_ok': metrics['mape'] < 15
        }
    
    def log_mlflow_run(self, metrics: Dict[str, float], objectives: Dict[str, bool], 
                       params: Dict[str, Any] = None, X_train=None, X_val=None) -> None:
        """Log experiment to MLflow"""
        with mlflow.start_run(run_name=f"{self.model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log basic parameters
            mlflow.log_param("model_type", self.model_name)
            if X_train is not None:
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("n_features", X_train.shape[1])
            if X_val is not None:
                mlflow.log_param("val_size", len(X_val))
            
            # Log model-specific parameters
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log objectives met
            mlflow.log_metric("objectives_met", sum(objectives.values()))
            
            # Log model
            if self.model is not None:
                mlflow.sklearn.log_model(self.model, "model")
    
    def print_metrics(self, metrics: Dict[str, float], objectives: Dict[str, bool], 
                     dataset_name: str = "Dataset") -> None:
        """Print formatted metrics"""
        print(f"\n📊 Métricas para {self.model_name} - {dataset_name}:")
        print(f"MAE: {metrics['mae']:.2f} (objetivo: <400) {'✅' if objectives['mae_ok'] else '❌'}")
        print(f"RMSE: {metrics['rmse']:.2f} (objetivo: <600) {'✅' if objectives['rmse_ok'] else '❌'}")
        print(f"R²: {metrics['r2']:.4f} (objetivo: >0.85) {'✅' if objectives['r2_ok'] else '❌'}")
        print(f"MAPE: {metrics['mape']:.2f}% (objetivo: <15%) {'✅' if objectives['mape_ok'] else '❌'}")
        print(f"Objetivos cumplidos: {sum(objectives.values())}/4")


def setup_mlflow(experiment_name: str = "bike_sharing_demand_prediction") -> None:
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    print(f"MLflow configurado - Experimento: {experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")