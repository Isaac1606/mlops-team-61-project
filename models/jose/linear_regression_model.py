# Linear Regression Model for Bike Sharing Demand Prediction

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
from .base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """
    Linear Regression model implementation for bike sharing demand prediction.
    This serves as a baseline model.
    """
    
    def __init__(self):
        super().__init__("LinearRegression")
        self.scaler = StandardScaler()
        self.use_scaling = True
        
    def create_model(self, **params) -> LinearRegression:
        """Create Linear Regression model"""
        default_params = {
            'fit_intercept': True,
            'copy_X': True
        }
        default_params.update(params)
        
        self.model = LinearRegression(**default_params)
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Train the Linear Regression model"""
        print(f"🚀 Entrenando {self.model_name}...")
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on training data
        y_train_pred = self.predict(X_train)
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        train_objectives = self.evaluate_objectives(train_metrics)
        
        self.print_metrics(train_metrics, train_objectives, "Entrenamiento")
        
        # Evaluate on validation data if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            val_metrics = self.calculate_metrics(y_val, y_val_pred)
            val_objectives = self.evaluate_objectives(val_metrics)
            
            self.print_metrics(val_metrics, val_objectives, "Validación")
            
            # Log to MLflow
            params = {
                'fit_intercept': self.model.fit_intercept,
                'normalize': False,
                'copy_X': True,
                'scaling': self.use_scaling
            }
            
            self.log_mlflow_run(val_metrics, val_objectives, params, X_train, X_val)
            
            return val_metrics, val_objectives
        
        return train_metrics, train_objectives
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with scaling"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature coefficients as importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df