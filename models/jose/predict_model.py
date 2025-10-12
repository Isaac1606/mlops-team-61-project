# Model Evaluation and Prediction Script

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.paths_config import PROJECT_ROOT


class BikeSharePredictor:
    """
    Model predictor for bike sharing demand prediction.
    Loads trained models and makes predictions on new data.
    """
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = str(Path(__file__).parent)  # models/jose/
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_list = None
        self.model_name = None
        self.is_loaded = False
        
    def load_model(self, model_name: str = "best") -> None:
        """Load trained model and artifacts"""
        print(f"📦 Cargando modelo desde {self.models_dir}...")
        
        try:
            # Load feature list
            feature_path = f"{self.models_dir}/feature_list.pkl"
            self.feature_list = joblib.load(feature_path)
            print(f"✅ Features cargadas: {len(self.feature_list)} características")
            
            # Find and load best model
            if model_name == "best":
                # Try to find best model file
                import os
                model_files = [f for f in os.listdir(self.models_dir) if f.startswith("best_model_") and f.endswith(".pkl")]
                if not model_files:
                    raise FileNotFoundError("No se encontró modelo 'best_model_'")
                
                model_file = model_files[0]
                self.model_name = model_file.replace("best_model_", "").replace(".pkl", "")
            else:
                model_file = f"best_model_{model_name.lower()}.pkl"
                self.model_name = model_name
            
            # Load model
            model_path = f"{self.models_dir}/{model_file}"
            self.model = joblib.load(model_path)
            print(f"✅ Modelo cargado: {self.model_name}")
            
            # Load scaler if exists (for LinearRegression)
            scaler_path = f"{self.models_dir}/scaler.pkl"
            try:
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler cargado")
            except FileNotFoundError:
                print("ℹ️  No se encontró scaler (normal para Random Forest/XGBoost)")
            
            self.is_loaded = True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecuta load_model() primero.")
        
        df_prep = df.copy()
        
        # Feature Engineering (same as training)
        df_prep['hr_sin'] = np.sin(2 * np.pi * df_prep['hr'] / 24)
        df_prep['hr_cos'] = np.cos(2 * np.pi * df_prep['hr'] / 24)
        df_prep['mnth_sin'] = np.sin(2 * np.pi * df_prep['mnth'] / 12)
        df_prep['mnth_cos'] = np.cos(2 * np.pi * df_prep['mnth'] / 12)
        df_prep['temp_season'] = df_prep['temp'] * df_prep['season']
        df_prep['hr_workingday'] = df_prep['hr'] * df_prep['workingday']
        
        # Select only required features
        X = df_prep[self.feature_list]
        
        # Apply scaling if needed
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X), 
                columns=self.feature_list, 
                index=X.index
            )
        
        return X
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecuta load_model() primero.")
        
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_single(self, season: int, yr: int, mnth: int, hr: int, 
                      holiday: int, weekday: int, workingday: int, 
                      weathersit: int, temp: float, atemp: float, 
                      hum: float, windspeed: float) -> float:
        """Make prediction for a single observation"""
        
        # Create DataFrame with single row
        data = pd.DataFrame({
            'season': [season],
            'yr': [yr],
            'mnth': [mnth],
            'hr': [hr],
            'holiday': [holiday],
            'weekday': [weekday],
            'workingday': [workingday],
            'weathersit': [weathersit],
            'temp': [temp],
            'atemp': [atemp],
            'hum': [hum],
            'windspeed': [windspeed]
        })
        
        prediction = self.predict(data)
        return prediction[0]
    
    def evaluate_model(self, test_data_path: str) -> Dict[str, float]:
        """Evaluate loaded model on test data"""
        print(f"🔬 Evaluando modelo en datos de test...")
        
        # Load test data
        df_test = pd.read_csv(test_data_path)
        
        # Prepare features and target
        X_test = self.prepare_features(df_test)
        y_test = df_test['cnt']
        
        # Make predictions
        y_pred = self.model.predict(X_test if self.scaler is None else X_test.values)
        y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        # Print results
        print(f"\n📊 Métricas de evaluación:")
        print(f"MAE: {mae:.2f} (objetivo: <400)")
        print(f"RMSE: {rmse:.2f} (objetivo: <600)")
        print(f"R²: {r2:.4f} (objetivo: >0.85)")
        print(f"MAPE: {mape:.2f}% (objetivo: <15%)")
        
        # Check objectives
        objectives_met = sum([
            mae < 400,
            rmse < 600,
            r2 > 0.85,
            mape < 15
        ])
        print(f"Objetivos cumplidos: {objectives_met}/4")
        
        return metrics
    
    def plot_predictions(self, test_data_path: str) -> None:
        """Plot prediction results"""
        # Load test data
        df_test = pd.read_csv(test_data_path)
        X_test = self.prepare_features(df_test)
        y_test = df_test['cnt']
        
        # Make predictions
        y_pred = self.model.predict(X_test if self.scaler is None else X_test.values)
        y_pred = np.maximum(y_pred, 0)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scatter plot
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Valores Reales')
        axes[0, 0].set_ylabel('Predicciones')
        axes[0, 0].set_title(f'{self.model_name}: Predicciones vs Reales')
        
        # Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicciones')
        axes[0, 1].set_ylabel('Residuos')
        axes[0, 1].set_title('Análisis de Residuos')
        
        # Time series (first 200 points)
        n_points = min(200, len(y_test))
        axes[1, 0].plot(range(n_points), y_test.iloc[:n_points], label='Real', alpha=0.7)
        axes[1, 0].plot(range(n_points), y_pred[:n_points], label='Predicción', alpha=0.7)
        axes[1, 0].set_xlabel('Tiempo')
        axes[1, 0].set_ylabel('Demanda')
        axes[1, 0].set_title('Serie Temporal (primeros 200 puntos)')
        axes[1, 0].legend()
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Error (Real - Predicción)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Errores')
        
        plt.tight_layout()
        plt.show()


def predict_demand_example():
    """Example of how to use the predictor"""
    print("🎯 EJEMPLO DE PREDICCIÓN DE DEMANDA")
    print("="*50)
    
    # Initialize predictor
    predictor = BikeSharePredictor()
    
    try:
        # Load model
        predictor.load_model()
        
        # Example prediction for a specific scenario
        print("\n📝 Predicción para escenario específico:")
        print("Día laboral de verano, 8:00 AM, temperatura agradable")
        
        prediction = predictor.predict_single(
            season=3,      # Summer
            yr=1,          # 2012
            mnth=7,        # July
            hr=8,          # 8 AM
            holiday=0,     # Not holiday
            weekday=1,     # Tuesday
            workingday=1,  # Working day
            weathersit=1,  # Clear weather
            temp=0.6,      # Normalized temperature (~24°C)
            atemp=0.6,     # Normalized feeling temp
            hum=0.6,       # 60% humidity
            windspeed=0.2  # Light wind
        )
        
        print(f"🚴 Demanda predicha: {prediction:.0f} bicicletas/hora")
        
        # Try with different scenarios
        scenarios = [
            {
                'name': 'Día laboral - Hora pico mañana',
                'params': {'season': 3, 'yr': 1, 'mnth': 7, 'hr': 8, 'holiday': 0, 
                          'weekday': 1, 'workingday': 1, 'weathersit': 1, 'temp': 0.6, 
                          'atemp': 0.6, 'hum': 0.6, 'windspeed': 0.2}
            },
            {
                'name': 'Fin de semana - Mediodía',
                'params': {'season': 3, 'yr': 1, 'mnth': 7, 'hr': 12, 'holiday': 0, 
                          'weekday': 6, 'workingday': 0, 'weathersit': 1, 'temp': 0.7, 
                          'atemp': 0.7, 'hum': 0.5, 'windspeed': 0.1}
            },
            {
                'name': 'Día lluvioso - Tarde',
                'params': {'season': 2, 'yr': 1, 'mnth': 4, 'hr': 17, 'holiday': 0, 
                          'weekday': 3, 'workingday': 1, 'weathersit': 3, 'temp': 0.4, 
                          'atemp': 0.4, 'hum': 0.8, 'windspeed': 0.4}
            }
        ]
        
        print(f"\n📊 Comparación de escenarios:")
        for scenario in scenarios:
            pred = predictor.predict_single(**scenario['params'])
            print(f"{scenario['name']}: {pred:.0f} bicicletas/hora")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Asegúrate de haber entrenado los modelos primero ejecutando train_models.py")


if __name__ == "__main__":
    predict_demand_example()