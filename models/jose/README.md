# Machine Learning Models - Bike Sharing Demand Prediction

Este directorio contiene la implementación completa de modelos de Machine Learning para predecir la demanda de bicicletas compartidas, con integración completa de MLflow para gestión de experimentos y modelos.

## 📁 Estructura de Archivos

```
src/models/
├── __init__.py                 # Inicialización del paquete
├── base_model.py              # Clase base para todos los modelos
├── linear_regression_model.py # Modelo de regresión lineal (baseline)
├── random_forest_model.py     # Modelo Random Forest
├── xgboost_model.py          # Modelo XGBoost (principal)
├── train_models.py           # Script principal de entrenamiento
├── predict_model.py          # Script de predicción e inferencia
├── mlflow_manager.py         # Gestión de experimentos y modelos MLflow
├── test_models.py            # Pruebas rápidas de funcionalidad
└── README.md                 # Esta documentación
```

## 🎯 Objetivos de Negocio

Los modelos están diseñados para cumplir los siguientes objetivos:

- **MAE** < 400 bicicletas/hora
- **RMSE** < 600
- **MAPE** < 15%
- **R²** > 0.85

## 🚀 Uso Rápido

### 1. Entrenar Modelos Completos

```bash
# Entrenamiento completo con optimización de hiperparámetros
python src/models/train_models.py
```

### 2. Hacer Predicciones

```bash
# Ejemplo de predicción
python src/models/predict_model.py
```

### 3. Gestionar Experimentos MLflow

```bash
# Gestión de modelos y experimentos
python src/models/mlflow_manager.py

# Abrir interfaz web de MLflow
mlflow ui --backend-store-uri file:./mlruns
```

### 4. Prueba Rápida

```bash
# Verificar que todo funciona correctamente
python src/models/test_models.py
```

## 📊 Modelos Implementados

### 1. Linear Regression (Baseline)
- **Archivo**: `linear_regression_model.py`
- **Propósito**: Modelo baseline simple
- **Características**: Normalización automática, interpretabilidad alta
- **Uso**: Comparación y línea base de rendimiento

### 2. Random Forest
- **Archivo**: `random_forest_model.py`
- **Propósito**: Modelo ensemble robusto
- **Características**: GridSearchCV, feature importance, sin normalización requerida
- **Hiperparámetros optimizados**: n_estimators, max_depth, min_samples_split, min_samples_leaf

### 3. XGBoost (Modelo Principal)
- **Archivo**: `xgboost_model.py`
- **Propósito**: Modelo de gradient boosting optimizado
- **Características**: GridSearchCV, early stopping, feature importance
- **Hiperparámetros optimizados**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree

## 🔧 Características Técnicas

### Feature Engineering Automático
Todos los modelos incluyen automáticamente:
- **Features cíclicas**: sin/cos para hora y mes
- **Features de interacción**: temp×season, hr×workingday
- **Normalización**: Automática para Linear Regression

### División Temporal
- **Entrenamiento**: 80% (datos más antiguos)
- **Validación**: 10% (datos intermedios)
- **Test**: 10% (datos más recientes)

### Métricas de Evaluación
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coeficiente de determinación
- **MAPE**: Mean Absolute Percentage Error

## 📈 Integración MLflow

### Tracking Automático
- Parámetros del modelo
- Métricas de evaluación
- Artifacts (modelos entrenados)
- Feature importance
- Objetivos de negocio cumplidos

### Model Registry
- Registro automático de modelos
- Versionado de modelos
- Promoción de modelos (Staging → Production)
- Comparación de experimentos

### Artifacts Guardados
- Modelo entrenado (`.pkl`)
- Scaler (si aplica)
- Lista de features
- Métricas finales
- Resultados de entrenamiento

## 💾 Persistencia de Modelos

Los modelos entrenados se guardan en:
```
models/jose/
├── best_model_[nombre].pkl      # Mejor modelo entrenado
├── scaler.pkl                   # Scaler (solo Linear Regression)
├── feature_list.pkl            # Lista de características
├── training_results.pkl        # Resultados completos
└── mlflow_export/              # Exportación desde MLflow
```

## 🔮 Predicción e Inferencia

### Predicción Individual
```python
from src.models.predict_model import BikeSharePredictor

# El predictor ahora usa models/jose/ por defecto
predictor = BikeSharePredictor()
predictor.load_model()

demanda = predictor.predict_single(
    season=3, yr=1, mnth=7, hr=8, holiday=0,
    weekday=1, workingday=1, weathersit=1,
    temp=0.6, atemp=0.6, hum=0.6, windspeed=0.2
)
print(f"Demanda predicha: {demanda:.0f} bicicletas/hora")
```

### Predicción en Lote
```python
import pandas as pd

# Cargar datos nuevos
df_new = pd.read_csv("nuevos_datos.csv")

# Hacer predicciones
predicciones = predictor.predict(df_new)
```

## 🎛️ Configuración y Personalización

### Modificar Hiperparámetros
Edita los diccionarios `param_grid` en cada modelo:

```python
# En random_forest_model.py
param_grid = {
    'n_estimators': [100, 200, 300],  # Añadir más valores
    'max_depth': [10, 15, 20, None],
    # ... más parámetros
}
```

### Añadir Nuevas Métricas
Modifica el método `calculate_metrics` en `base_model.py`:

```python
def calculate_metrics(self, y_true, y_pred):
    # Métricas existentes...
    nueva_métrica = custom_metric(y_true, y_pred)
    return {
        # métricas existentes...
        'nueva_métrica': nueva_métrica
    }
```

### Cambiar Objetivos de Negocio
Modifica el método `evaluate_objectives` en `base_model.py`:

```python
def evaluate_objectives(self, metrics):
    return {
        'mae_ok': metrics['mae'] < 350,  # Nuevo objetivo
        'rmse_ok': metrics['rmse'] < 500,  # Nuevo objetivo
        # ... más objetivos
    }
```

## 🔄 Flujo de Trabajo Recomendado

1. **Preparación**: Asegurar que `data/interim/bike_sharing_clean.csv` existe
2. **Entrenamiento**: Ejecutar `train_models.py` con grid search
3. **Evaluación**: Revisar resultados en MLflow UI
4. **Registro**: Registrar mejor modelo en Model Registry
5. **Despliegue**: Usar `predict_model.py` para inferencia
6. **Monitoreo**: Comparar rendimiento con nuevos datos

## 🐛 Troubleshooting

### Error: "No module named 'xgboost'"
```bash
pip install xgboost
```

### Error: "Archivo de datos no encontrado"
Ejecuta el notebook EDA para generar `bike_sharing_clean.csv`

### Error: MLflow UI no abre
```bash
# Verificar puerto
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

### Error: "Model must be trained before making predictions"
Ejecuta primero `train_models.py` para generar modelos

## 📋 Dependencias

Ver `requirements.txt` para la lista completa. Principales:
- `scikit-learn`: Modelos base y métricas
- `xgboost`: Modelo XGBoost
- `mlflow`: Tracking y model registry
- `pandas`, `numpy`: Manipulación de datos
- `matplotlib`: Visualizaciones
- `joblib`: Persistencia de modelos

## 🎯 Próximos Pasos

- [ ] Implementar validación cruzada temporal
- [ ] Añadir modelos adicionales (LightGBM, CatBoost)
- [ ] Implementar feature selection automático
- [ ] Crear pipeline de reentrenamiento automático
- [ ] Implementar monitoring de data drift
- [ ] Crear API REST para inferencia
- [ ] Implementar A/B testing de modelos