# MLflow Model Management and Registry

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.paths_config import PROJECT_ROOT


class MLflowModelManager:
    """
    MLflow model management and registry for bike sharing models.
    Handles model registration, versioning, and deployment.
    """
    
    def __init__(self, tracking_uri: str = "file:./mlruns", 
                 experiment_name: str = "bike_sharing_demand_prediction"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self) -> None:
        """Setup MLflow tracking and experiment"""
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
        except:
            mlflow.create_experiment(self.experiment_name)
        
        mlflow.set_experiment(self.experiment_name)
        print(f"MLflow configurado - Experimento: {self.experiment_name}")
        print(f"Tracking URI: {self.tracking_uri}")
    
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments"""
        experiments = mlflow.search_experiments()
        return pd.DataFrame([{
            'experiment_id': exp.experiment_id,
            'name': exp.name,
            'lifecycle_stage': exp.lifecycle_stage,
            'artifact_location': exp.artifact_location
        } for exp in experiments])
    
    def list_runs(self, experiment_name: str = None, max_results: int = 100) -> pd.DataFrame:
        """List runs from experiment"""
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["metrics.mae ASC"]
        )
        
        return runs
    
    def get_best_run(self, metric: str = "mae", ascending: bool = True) -> mlflow.entities.Run:
        """Get best run based on metric"""
        runs_df = self.list_runs()
        
        if runs_df.empty:
            raise ValueError("No runs found in experiment")
        
        # Sort by metric
        metric_col = f"metrics.{metric}"
        if metric_col not in runs_df.columns:
            raise ValueError(f"Metric '{metric}' not found in runs")
        
        runs_df = runs_df.sort_values(metric_col, ascending=ascending)
        best_run_id = runs_df.iloc[0]['run_id']
        
        return mlflow.get_run(best_run_id)
    
    def register_model(self, run_id: str, model_name: str = "BikeShareDemandModel") -> None:
        """Register model in MLflow Model Registry"""
        try:
            # Get the model URI
            model_uri = f"runs:/{run_id}/model"
            
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            
            print(f"✅ Modelo registrado: {model_name}")
            print(f"Versión: {model_version.version}")
            print(f"Run ID: {run_id}")
            
            return model_version
            
        except Exception as e:
            print(f"❌ Error registrando modelo: {str(e)}")
            raise
    
    def promote_model(self, model_name: str, version: int, stage: str = "Production") -> None:
        """Promote model to specific stage"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"✅ Modelo {model_name} v{version} promovido a {stage}")
            
        except Exception as e:
            print(f"❌ Error promoviendo modelo: {str(e)}")
            raise
    
    def list_registered_models(self) -> List[Dict]:
        """List all registered models"""
        client = mlflow.tracking.MlflowClient()
        
        models = client.list_registered_models()
        
        model_info = []
        for model in models:
            latest_versions = model.latest_versions
            for version in latest_versions:
                model_info.append({
                    'name': model.name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'creation_timestamp': version.creation_timestamp,
                    'run_id': version.run_id
                })
        
        return model_info
    
    def load_model_from_registry(self, model_name: str, stage: str = "Production"):
        """Load model from registry"""
        model_uri = f"models:/{model_name}/{stage}"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Modelo cargado desde registry: {model_name} ({stage})")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo desde registry: {str(e)}")
            raise
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs"""
        comparison_data = []
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            
            comparison_data.append({
                'run_id': run_id,
                'model_type': run.data.params.get('model_type', 'Unknown'),
                'mae': run.data.metrics.get('mae', None),
                'rmse': run.data.metrics.get('rmse', None),
                'r2': run.data.metrics.get('r2', None),
                'mape': run.data.metrics.get('mape', None),
                'objectives_met': run.data.metrics.get('objectives_met', None),
                'start_time': run.info.start_time,
                'status': run.info.status
            })
        
        return pd.DataFrame(comparison_data)
    
    def create_model_report(self, run_id: str) -> Dict:
        """Create detailed model report"""
        run = mlflow.get_run(run_id)
        
        report = {
            'run_info': {
                'run_id': run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time
            },
            'parameters': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }
        
        # Check business objectives
        metrics = run.data.metrics
        if all(metric in metrics for metric in ['mae', 'rmse', 'r2', 'mape']):
            objectives = {
                'mae_ok': metrics['mae'] < 400,
                'rmse_ok': metrics['rmse'] < 600,
                'r2_ok': metrics['r2'] > 0.85,
                'mape_ok': metrics['mape'] < 15
            }
            report['objectives_analysis'] = objectives
            report['objectives_met'] = sum(objectives.values())
        
        return report
    
    def export_best_model(self, output_dir: str = None) -> str:
        """Export best model to local directory"""
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "models" / "jose" / "mlflow_export")
        
        # Get best run
        best_run = self.get_best_run()
        run_id = best_run.info.run_id
        
        # Download model
        model_uri = f"runs:/{run_id}/model"
        local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=output_dir)
        
        print(f"✅ Mejor modelo exportado a: {local_path}")
        print(f"Run ID: {run_id}")
        print(f"MAE: {best_run.data.metrics.get('mae', 'N/A')}")
        
        return local_path
    
    def start_mlflow_ui(self, port: int = 5000) -> None:
        """Instructions to start MLflow UI"""
        print(f"\n📊 Para abrir la interfaz de MLflow, ejecuta:")
        print(f"mlflow ui --backend-store-uri {self.tracking_uri} --port {port}")
        print(f"Luego visita: http://localhost:{port}")


def manage_models_example():
    """Example of MLflow model management"""
    print("🏗️ GESTIÓN DE MODELOS CON MLFLOW")
    print("="*50)
    
    # Initialize manager
    manager = MLflowModelManager()
    
    try:
        # List experiments
        print("\n📋 Experimentos disponibles:")
        experiments_df = manager.list_experiments()
        print(experiments_df)
        
        # List runs
        print(f"\n🏃 Runs del experimento '{manager.experiment_name}':")
        runs_df = manager.list_runs(max_results=10)
        if not runs_df.empty:
            # Show key columns
            display_cols = ['run_id', 'metrics.mae', 'metrics.rmse', 'metrics.r2', 'params.model_type']
            available_cols = [col for col in display_cols if col in runs_df.columns]
            print(runs_df[available_cols].head())
            
            # Get best run
            print(f"\n🏆 Mejor run (menor MAE):")
            try:
                best_run = manager.get_best_run()
                print(f"Run ID: {best_run.info.run_id}")
                print(f"MAE: {best_run.data.metrics.get('mae', 'N/A')}")
                print(f"Modelo: {best_run.data.params.get('model_type', 'N/A')}")
                
                # Create detailed report
                print(f"\n📊 Reporte detallado del mejor modelo:")
                report = manager.create_model_report(best_run.info.run_id)
                print(f"Objetivos cumplidos: {report.get('objectives_met', 'N/A')}/4")
                
            except Exception as e:
                print(f"Error obteniendo mejor run: {str(e)}")
        else:
            print("No hay runs disponibles. Ejecuta train_models.py primero.")
        
        # Show registered models
        print(f"\n📦 Modelos registrados:")
        registered_models = manager.list_registered_models()
        if registered_models:
            models_df = pd.DataFrame(registered_models)
            print(models_df)
        else:
            print("No hay modelos registrados aún.")
        
        # Instructions for UI
        manager.start_mlflow_ui()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    manage_models_example()