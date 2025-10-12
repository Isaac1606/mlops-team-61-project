# Model Trainer - Main script for training and evaluating models

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.paths_config import PROJECT_ROOT, RAW_FILE_PATH
from models.jose.base_model import setup_mlflow
from models.jose.linear_regression_model import LinearRegressionModel
from models.jose.random_forest_model import RandomForestModel
try:
    from models.jose.xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost no disponible. Instala con: pip install xgboost")
    XGBOOST_AVAILABLE = False

import joblib
import matplotlib.pyplot as plt


class BikeShareModelTrainer:
    """
    Main trainer class for bike sharing demand prediction models.
    Handles data preparation, model training, evaluation, and comparison.
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or str(PROJECT_ROOT / "data" / "interim" / "bike_sharing_clean.csv")
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_cols = []
        
        # Setup MLflow
        setup_mlflow()
        
    def load_and_prepare_data(self) -> None:
        """Load and prepare data for model training"""
        print("📋 Cargando y preparando datos...")
        
        # Load clean data
        df = pd.read_csv(self.data_path)
        print(f"Datos cargados: {df.shape}")
        
        # Feature Engineering
        print("🔧 Realizando feature engineering...")
        
        # Cyclical features for hour
        df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
        df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
        
        # Cyclical features for month
        df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
        df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
        
        # Interaction features
        df['temp_season'] = df['temp'] * df['season']
        df['hr_workingday'] = df['hr'] * df['workingday']
        
        # Select features
        self.feature_cols = [
            'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
            'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'temp_season', 'hr_workingday'
        ]
        
        X = df[self.feature_cols]
        y = df['cnt']
        
        # Temporal split (80% train, 10% val, 10% test)
        df_sorted = df.sort_values('dteday').reset_index(drop=True)
        n_total = len(df_sorted)
        n_train = int(0.8 * n_total)
        n_val = int(0.9 * n_total)
        
        self.X_train = df_sorted[self.feature_cols].iloc[:n_train]
        self.X_val = df_sorted[self.feature_cols].iloc[n_train:n_val]
        self.X_test = df_sorted[self.feature_cols].iloc[n_val:]
        
        self.y_train = df_sorted['cnt'].iloc[:n_train]
        self.y_val = df_sorted['cnt'].iloc[n_train:n_val]
        self.y_test = df_sorted['cnt'].iloc[n_val:]
        
        print(f"✅ Datos preparados:")
        print(f"  - Train: {self.X_train.shape}")
        print(f"  - Validation: {self.X_val.shape}")
        print(f"  - Test: {self.X_test.shape}")
        print(f"  - Features: {len(self.feature_cols)}")
        
    def train_models(self, use_grid_search: bool = True) -> None:
        """Train all available models"""
        print("\n🚀 Iniciando entrenamiento de modelos...")
        
        # Initialize models
        self.models = {
            'LinearRegression': LinearRegressionModel(),
            'RandomForest': RandomForestModel()
        }
        
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBoostModel()
        
        # Train each model
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Entrenando {name}")
            print(f"{'='*60}")
            
            try:
                if name == 'LinearRegression':
                    metrics, objectives = model.train(self.X_train, self.y_train, self.X_val, self.y_val)
                else:
                    metrics, objectives = model.train(
                        self.X_train, self.y_train, self.X_val, self.y_val, 
                        use_grid_search=use_grid_search
                    )
                
                self.results[name] = {
                    'metrics': metrics,
                    'objectives': objectives,
                    'model': model
                }
                
                print(f"✅ {name} entrenado exitosamente")
                
            except Exception as e:
                print(f"❌ Error entrenando {name}: {str(e)}")
                continue
    
    def compare_models(self) -> None:
        """Compare all trained models"""
        print(f"\n📊 COMPARACIÓN DE MODELOS")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in self.results.items():
            metrics = result['metrics']
            objectives = result['objectives']
            
            comparison_data.append({
                'Modelo': name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'MAPE (%)': metrics['mape'],
                'Objetivos': f"{sum(objectives.values())}/4"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_idx = comparison_df['MAE'].idxmin()
        self.best_model_name = comparison_df.loc[best_idx, 'Modelo']
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🏆 MEJOR MODELO: {self.best_model_name}")
        
        # Check business objectives
        best_metrics = self.results[self.best_model_name]['metrics']
        print(f"\n🎯 OBJETIVOS DE NEGOCIO:")
        print(f"MAE < 400: {'✅' if best_metrics['mae'] < 400 else '❌'} ({best_metrics['mae']:.2f})")
        print(f"RMSE < 600: {'✅' if best_metrics['rmse'] < 600 else '❌'} ({best_metrics['rmse']:.2f})")
        print(f"R² > 0.85: {'✅' if best_metrics['r2'] > 0.85 else '❌'} ({best_metrics['r2']:.4f})")
        print(f"MAPE < 15%: {'✅' if best_metrics['mape'] < 15 else '❌'} ({best_metrics['mape']:.2f}%)")
        
        return comparison_df
    
    def evaluate_final_model(self) -> None:
        """Evaluate best model on test set"""
        if self.best_model is None:
            raise ValueError("No hay modelo seleccionado. Ejecuta compare_models() primero.")
        
        print(f"\n🔬 EVALUACIÓN FINAL EN TEST SET")
        print("="*50)
        
        # Make predictions on test set
        y_test_pred = self.best_model.predict(self.X_test)
        
        # Calculate final metrics
        final_metrics = self.best_model.calculate_metrics(self.y_test, y_test_pred)
        final_objectives = self.best_model.evaluate_objectives(final_metrics)
        
        self.best_model.print_metrics(final_metrics, final_objectives, "TEST")
        
        # Plot results
        self.plot_predictions(self.y_test, y_test_pred)
        
        return final_metrics, final_objectives
    
    def plot_predictions(self, y_true, y_pred) -> None:
        """Plot prediction results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Valores Reales')
        axes[0].set_ylabel('Predicciones')
        axes[0].set_title(f'{self.best_model_name}: Predicciones vs Reales')
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicciones')
        axes[1].set_ylabel('Residuos')
        axes[1].set_title('Análisis de Residuos')
        
        # Histogram of residuals
        axes[2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Residuos')
        axes[2].set_ylabel('Frecuencia')
        axes[2].set_title('Distribución de Residuos')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, models_dir: str = None) -> None:
        """Save trained models and artifacts"""
        if models_dir is None:
            models_dir = str(Path(__file__).parent)  # models/jose/
        
        os.makedirs(models_dir, exist_ok=True)
        
        print(f"\n💾 Guardando modelos en {models_dir}...")
        
        # Save best model
        if self.best_model is not None:
            model_path = os.path.join(models_dir, f"best_model_{self.best_model_name.lower()}.pkl")
            joblib.dump(self.best_model.model, model_path)
            print(f"✅ Mejor modelo guardado: {model_path}")
            
            # Save scaler if it's Linear Regression
            if hasattr(self.best_model, 'scaler'):
                scaler_path = os.path.join(models_dir, "scaler.pkl")
                joblib.dump(self.best_model.scaler, scaler_path)
                print(f"✅ Scaler guardado: {scaler_path}")
        
        # Save feature list
        feature_path = os.path.join(models_dir, "feature_list.pkl")
        joblib.dump(self.feature_cols, feature_path)
        print(f"✅ Lista de features guardada: {feature_path}")
        
        # Save results
        results_path = os.path.join(models_dir, "training_results.pkl")
        joblib.dump(self.results, results_path)
        print(f"✅ Resultados guardados: {results_path}")
    
    def run_complete_training(self, use_grid_search: bool = True) -> None:
        """Run complete training pipeline"""
        print("🎯 INICIANDO PIPELINE COMPLETO DE ENTRENAMIENTO")
        print("="*70)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train models
        self.train_models(use_grid_search=use_grid_search)
        
        # Compare models
        self.compare_models()
        
        # Final evaluation
        self.evaluate_final_model()
        
        # Save models
        self.save_models()
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
        print(f"Mejor modelo: {self.best_model_name}")
        print(f"Modelos guardados en: {PROJECT_ROOT / 'models' / 'jose'}")
        print(f"Experimentos MLflow en: ./mlruns")


if __name__ == "__main__":
    # Run training
    trainer = BikeShareModelTrainer()
    trainer.run_complete_training(use_grid_search=True)