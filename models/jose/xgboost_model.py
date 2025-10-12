# XGBoost Model for Bike Sharing Demand Prediction

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost model implementation for bike sharing demand prediction.
    Expected to be the primary model with best performance.
    """
    
    def __init__(self):
        super().__init__("XGBoost")
        self.best_params = None
        self.grid_search = None
        
    def create_model(self, **params) -> xgb.XGBRegressor:
        """Create XGBoost model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        self.model = xgb.XGBRegressor(**default_params)
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              use_grid_search: bool = True) -> None:
        """Train the XGBoost model with optional hyperparameter tuning"""
        print(f"🚀 Entrenando {self.model_name}...")
        
        if use_grid_search:
            print("🔍 Optimizando hiperparámetros con GridSearchCV...")
            
            # Define parameter grid (smaller grid for faster execution)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Create base model
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
            # GridSearch with cross-validation
            self.grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3, 
                scoring='neg_mean_absolute_error', 
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            self.grid_search.fit(X_train, y_train)
            
            # Get best model
            self.model = self.grid_search.best_estimator_
            self.best_params = self.grid_search.best_params_
            
            print(f"✅ Mejores parámetros: {self.best_params}")
        else:
            # Create and train model with default parameters
            if self.model is None:
                self.create_model()
            
            # Train with early stopping if validation data is provided
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        
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
            params = self.best_params if self.best_params else {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'learning_rate': self.model.learning_rate,
                'subsample': self.model.subsample,
                'colsample_bytree': self.model.colsample_bytree
            }
            params['grid_search'] = use_grid_search
            
            self.log_mlflow_run(val_metrics, val_objectives, params, X_train, X_val)
            
            return val_metrics, val_objectives
        
        return train_metrics, train_objectives
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance from XGBoost"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_importance(self, max_num_features: int = 20):
        """Plot feature importance using XGBoost's built-in plotting"""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting importance")
        
        xgb.plot_importance(self.model, max_num_features=max_num_features)
        
    def get_grid_search_results(self) -> pd.DataFrame:
        """Get results from grid search if performed"""
        if self.grid_search is None:
            raise ValueError("Grid search was not performed")
        
        results_df = pd.DataFrame(self.grid_search.cv_results_)
        return results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values('rank_test_score')