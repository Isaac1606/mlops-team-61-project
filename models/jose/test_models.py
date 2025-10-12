#!/usr/bin/env python3
"""
Quick test and demonstration of the bike sharing ML models
Run this script to test the model implementations
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from models.jose.train_models import BikeShareModelTrainer
    from models.jose.predict_model import BikeSharePredictor
    from models.jose.mlflow_manager import MLflowModelManager
    print("✅ Todas las importaciones exitosas")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)


def test_models():
    """Test model implementations"""
    print("🧪 PRUEBA RÁPIDA DE MODELOS")
    print("="*50)
    
    # Check if clean data exists
    data_path = PROJECT_ROOT / "data" / "interim" / "bike_sharing_clean.csv"
    if not data_path.exists():
        print(f"❌ Archivo de datos no encontrado: {data_path}")
        print("Asegúrate de haber ejecutado el notebook para generar los datos limpios")
        return
    
    print(f"✅ Datos encontrados: {data_path}")
    
    # Test quick training (without grid search for speed)
    print(f"\n🚀 Prueba rápida de entrenamiento (sin grid search)...")
    try:
        trainer = BikeShareModelTrainer(str(data_path))
        trainer.load_and_prepare_data()
        trainer.train_models(use_grid_search=False)  # Quick test
        trainer.compare_models()
        
        print(f"✅ Entrenamiento de prueba exitoso")
        print(f"Mejor modelo: {trainer.best_model_name}")
        
        # Save test models
        trainer.save_models()
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return
    
    # Test prediction
    print(f"\n🔮 Prueba de predicción...")
    try:
        predictor = BikeSharePredictor()
        predictor.load_model()
        
        # Test single prediction
        test_pred = predictor.predict_single(
            season=3, yr=1, mnth=7, hr=8, holiday=0,
            weekday=1, workingday=1, weathersit=1,
            temp=0.6, atemp=0.6, hum=0.6, windspeed=0.2
        )
        
        print(f"✅ Predicción de prueba: {test_pred:.0f} bicicletas/hora")
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return
    
    # Test MLflow manager
    print(f"\n📊 Prueba de MLflow Manager...")
    try:
        manager = MLflowModelManager()
        runs_df = manager.list_runs(max_results=5)
        print(f"✅ MLflow Manager funcionando. Runs encontrados: {len(runs_df)}")
        
    except Exception as e:
        print(f"❌ Error en MLflow Manager: {e}")
        return
    
    print(f"\n🎉 TODAS LAS PRUEBAS EXITOSAS")
    print(f"Los modelos están listos para usar en producción")
    print(f"\nPara entrenar modelos completos:")
    print(f"python src/models/train_models.py")
    print(f"\nPara gestionar modelos:")
    print(f"python src/models/mlflow_manager.py")
    print(f"\nPara hacer predicciones:")
    print(f"python src/models/predict_model.py")


if __name__ == "__main__":
    test_models()