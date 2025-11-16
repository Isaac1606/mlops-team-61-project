# Data Drift Detection and Performance Monitoring

## 📋 Tabla de Contenidos

1. [Quick Start](#quick-start)
2. [Introducción](#introducción)
3. [¿Qué es Data Drift?](#qué-es-data-drift)
4. [Arquitectura del Sistema](#arquitectura-del-sistema)
5. [Componentes Principales](#componentes-principales)
6. [Uso del Sistema](#uso-del-sistema)
7. [Ejemplos Prácticos](#ejemplos-prácticos)
8. [Tests y Validación](#tests-y-validación)
9. [Mejores Prácticas](#mejores-prácticas)
10. [Solución de Problemas](#solución-de-problemas)

---

## Quick Start

### Ejecutar Tests de Data Drift

```bash
# Todos los tests de drift
make test-drift

# O con pytest
pytest tests/data_drift/ -v
```

### Uso Básico en Código

```python
from src.models.data_drift import DataDriftDetector, PerformanceMonitor
import pandas as pd

# 1. Inicializar detector con datos de entrenamiento
detector = DataDriftDetector(X_train)

# 2. Detectar drift en datos de producción
drift_results = detector.detect_drift(X_production)

# 3. Verificar resultados
if drift_results['has_drift']:
    print(f"⚠️ Drift detectado! Score: {drift_results['drift_score']:.3f}")
    print(f"Features afectadas: {drift_results['summary']['drifted_features']}")

# 4. Monitorear performance
baseline_metrics = {'mae': 100.0, 'rmse': 150.0, 'r2': 0.85}
monitor = PerformanceMonitor(baseline_metrics, metric_type='mae')

perf_results = monitor.check_performance(y_production, y_pred_production)
if perf_results['has_degradation']:
    print(f"⚠️ Degradación: {perf_results['degradation_score']*100:.1f}%")
```

### Documentación Completa

Para más detalles, lee las secciones siguientes o consulta los ejemplos en `tests/data_drift/test_data_drift.py`.

---

## Introducción

Este proyecto incluye un sistema completo de detección de **Data Drift** y monitoreo de **degradación de performance** diseñado para mantener modelos de Machine Learning en producción. El sistema permite detectar cambios en la distribución de los datos y alertar cuando el rendimiento del modelo cae por debajo de los umbrales aceptables.

### Características Principales

- ✅ **Detección Estadística de Drift**: Tests de Kolmogorov-Smirnov, Chi-square y PSI
- ✅ **Monitoreo de Performance**: Detección automática de degradación
- ✅ **Generación de Datos Sintéticos**: Para pruebas y validación
- ✅ **Alertas Configurables**: Umbrales personalizables por métrica
- ✅ **Soporte Multi-tipo**: Features continuas y categóricas

---

## ¿Qué es Data Drift?

**Data Drift** (o "deriva de datos") ocurre cuando la distribución de los datos en producción difiere significativamente de los datos de entrenamiento. Esto puede deberse a:

- Cambios en el comportamiento del usuario
- Cambios estacionales
- Errores en el pipeline de datos
- Cambios en el entorno operativo
- Evolución natural del dominio

### Tipos de Drift

1. **Covariate Shift**: Cambio en la distribución de las features (X)
2. **Label Shift**: Cambio en la distribución del target (y)
3. **Concept Drift**: Cambio en la relación X → y

Nuestro sistema detecta principalmente **Covariate Shift** mediante tests estadísticos.

---

## Arquitectura del Sistema

El sistema de detección de drift está implementado en `src/models/data_drift.py` y consta de dos clases principales:

```
DataDriftDetector
├── Detección de drift en features
│   ├── Kolmogorov-Smirnov (features continuas)
│   ├── Chi-square (features categóricas)
│   └── PSI - Population Stability Index
├── Generación de datos sintéticos
│   ├── Mean shift drift
│   ├── Variance shift drift
│   └── Distribution shift drift
└── Reportes detallados por feature

PerformanceMonitor
├── Comparación con baseline
├── Cálculo de degradación
├── Alertas configurables
└── Soporte para MAE, RMSE, R²
```

---

## Componentes Principales

### 1. DataDriftDetector

Detecta cambios en la distribución de datos entre datos de referencia (entrenamiento) y datos actuales (producción).

#### Inicialización

```python
from src.models.data_drift import DataDriftDetector

# Usar todos los features numéricos por defecto
detector = DataDriftDetector(
    reference_data=X_train,
    threshold=0.05,  # P-value threshold para tests estadísticos
    psi_threshold=0.25  # Umbral PSI para drift significativo
)

# O especificar features manualmente
detector = DataDriftDetector(
    reference_data=X_train,
    feature_columns=['feature1', 'feature2', 'feature3'],
    categorical_columns=['category1', 'category2'],
    threshold=0.05,
    psi_threshold=0.25
)
```

#### Parámetros

- **reference_data** (pd.DataFrame): Datos de referencia (entrenamiento)
- **feature_columns** (List[str], opcional): Features continuas a monitorear
- **categorical_columns** (List[str], opcional): Features categóricas a monitorear
- **threshold** (float, default=0.05): P-value threshold para tests estadísticos
- **psi_threshold** (float, default=0.25): Umbral PSI para drift significativo

#### Detección de Drift

```python
# Detectar drift en datos de producción
results = detector.detect_drift(
    current_data=X_production,
    return_details=True  # Incluir información detallada por feature
)

# Estructura del resultado
{
    'has_drift': bool,           # ¿Se detectó drift?
    'drift_score': float,        # Score general de drift (0-1+)
    'feature_drifts': {          # Detalles por feature
        'feature1': {
            'type': 'continuous',
            'has_drift': bool,
            'ks_statistic': float,
            'ks_pvalue': float,
            'psi': float,
            'ref_mean': float,
            'curr_mean': float,
            ...
        },
        ...
    },
    'summary': {
        'total_features': int,
        'drifted_features': int,
        'tests_performed': int
    }
}
```

#### Tests Estadísticos

**Para Features Continuas:**
- **Kolmogorov-Smirnov Test**: Compara distribuciones empíricas
  - P-value < threshold → Drift detectado
- **PSI (Population Stability Index)**:
  - PSI < 0.1: Sin cambio significativo
  - 0.1 ≤ PSI < 0.25: Cambio moderado
  - PSI ≥ 0.25: Cambio significativo (drift)

**Para Features Categóricas:**
- **Chi-square Test**: Compara distribuciones de categorías
  - P-value < threshold → Drift detectado

### 2. PerformanceMonitor

Monitorea la degradación del rendimiento del modelo comparando métricas actuales con una baseline.

#### Inicialización

```python
from src.models.data_drift import PerformanceMonitor

# Métricas baseline (obtenidas en validación/entrenamiento)
baseline_metrics = {
    'mae': 100.0,
    'rmse': 150.0,
    'r2': 0.85
}

monitor = PerformanceMonitor(
    baseline_metrics=baseline_metrics,
    performance_threshold=0.2,  # 20% de degradación aceptable
    metric_type='mae'  # 'mae', 'rmse', o 'r2'
)
```

#### Parámetros

- **baseline_metrics** (Dict[str, float]): Métricas de referencia
- **performance_threshold** (float, default=0.2): Umbral de degradación relativa (20% = 0.2)
- **metric_type** (str): Métrica principal a monitorear ('mae', 'rmse', 'r2')

#### Monitoreo de Performance

```python
# Evaluar performance actual
perf_results = monitor.check_performance(
    y_true=y_production,
    y_pred=y_pred_production
)

# Estructura del resultado
{
    'has_degradation': bool,     # ¿Hay degradación?
    'degradation_score': float,  # Score de degradación (puede ser negativo si mejoró)
    'current_metrics': {         # Métricas actuales
        'mae': float,
        'rmse': float,
        'r2': float
    },
    'baseline_metrics': {...},   # Métricas de referencia
    'alert': bool                # ¿Debería alertar? (degradación > threshold * 1.5)
}
```

---

## Uso del Sistema

### Flujo Completo de Monitoreo

```python
from src.models.data_drift import DataDriftDetector, PerformanceMonitor
from src.models.model_evaluator import ModelEvaluator
from src.config.config_loader import ConfigLoader

# 1. Cargar configuración
config = ConfigLoader()

# 2. Inicializar detector con datos de entrenamiento
detector = DataDriftDetector(X_train)

# 3. Detectar drift en datos de producción
drift_results = detector.detect_drift(X_production)

# 4. Si hay drift, evaluar impacto en performance
if drift_results['has_drift']:
    print(f"⚠️ Drift detectado! Score: {drift_results['drift_score']:.3f}")
    print(f"Features afectadas: {drift_results['summary']['drifted_features']}")
    
    # Obtener predicciones del modelo
    y_pred = model.predict(X_production)
    
    # Monitorear performance
    evaluator = ModelEvaluator(config)
    current_metrics = evaluator.evaluate(y_production, y_pred)
    
    # Comparar con baseline
    baseline_metrics = {
        'mae': 100.0,  # Obtenido durante entrenamiento
        'rmse': 150.0,
        'r2': 0.85
    }
    
    monitor = PerformanceMonitor(
        baseline_metrics=baseline_metrics,
        performance_threshold=0.2,
        metric_type='mae'
    )
    
    perf_results = monitor.check_performance(y_production, y_pred)
    
    if perf_results['has_degradation']:
        print(f"⚠️ Degradación detectada: {perf_results['degradation_score']*100:.1f}%")
        
        if perf_results['alert']:
            print("🚨 ALERTA: Degradación significativa detectada!")
            # Aquí podrías enviar notificación, retrenar modelo, etc.
```

### Integración en Pipeline de Producción

```python
def monitor_production_model(model, X_production, y_production, X_train, baseline_metrics):
    """
    Monitoreo completo para modelo en producción.
    
    Returns:
        dict: Resultados completos de monitoreo
    """
    # 1. Detectar drift
    detector = DataDriftDetector(X_train)
    drift_results = detector.detect_drift(X_production)
    
    # 2. Obtener predicciones
    y_pred = model.predict(X_production)
    
    # 3. Monitorear performance
    monitor = PerformanceMonitor(
        baseline_metrics=baseline_metrics,
        performance_threshold=0.2,
        metric_type='mae'
    )
    perf_results = monitor.check_performance(y_production, y_pred)
    
    # 4. Combinar resultados
    return {
        'drift_detected': drift_results['has_drift'],
        'drift_score': drift_results['drift_score'],
        'performance_degradation': perf_results['has_degradation'],
        'degradation_score': perf_results['degradation_score'],
        'alert': perf_results['alert'],
        'recommendation': _get_recommendation(drift_results, perf_results)
    }

def _get_recommendation(drift_results, perf_results):
    """Generar recomendación basada en resultados."""
    if perf_results['alert']:
        return "Retrenar modelo inmediatamente"
    elif drift_results['has_drift'] and perf_results['has_degradation']:
        return "Monitorear de cerca, considerar retrenamiento"
    elif drift_results['has_drift']:
        return "Drift detectado pero sin impacto en performance aún"
    else:
        return "Sistema funcionando normalmente"
```

---

## Ejemplos Prácticos

### Ejemplo 1: Detección Básica de Drift

```python
import pandas as pd
from src.models.data_drift import DataDriftDetector

# Datos de entrenamiento
X_train = pd.read_csv('data/processed/bike_sharing_features_train.csv')

# Datos de producción (nuevos datos)
X_production = pd.read_csv('data/production/latest_batch.csv')

# Inicializar detector
detector = DataDriftDetector(X_train, threshold=0.05)

# Detectar drift
results = detector.detect_drift(X_production)

# Analizar resultados
if results['has_drift']:
    print(f"Drift Score: {results['drift_score']:.3f}")
    print(f"Features con drift: {results['summary']['drifted_features']}")
    
    # Detalles por feature
    for feature, info in results['feature_drifts'].items():
        if info['has_drift']:
            print(f"\n{feature}:")
            print(f"  Type: {info['type']}")
            if info['type'] == 'continuous':
                print(f"  PSI: {info['psi']:.3f}")
                print(f"  Reference mean: {info['ref_mean']:.2f}")
                print(f"  Current mean: {info['curr_mean']:.2f}")
```

### Ejemplo 2: Monitoreo Continuo

```python
from src.models.data_drift import PerformanceMonitor
import json
from datetime import datetime

# Baseline (obtenido durante entrenamiento)
baseline_metrics = {
    'mae': 95.3,
    'rmse': 142.1,
    'r2': 0.87
}

monitor = PerformanceMonitor(
    baseline_metrics=baseline_metrics,
    performance_threshold=0.2,
    metric_type='mae'
)

def daily_performance_check(y_true, y_pred):
    """Función para ejecutar diariamente."""
    results = monitor.check_performance(y_true, y_pred)
    
    # Registrar resultados
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'degradation_detected': results['has_degradation'],
        'degradation_score': results['degradation_score'],
        'current_mae': results['current_metrics']['mae'],
        'baseline_mae': results['baseline_metrics']['mae'],
        'alert': results['alert']
    }
    
    # Guardar log
    with open('logs/performance_monitoring.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Enviar alerta si es necesario
    if results['alert']:
        send_alert(log_entry)
    
    return results

# Uso diario
daily_performance_check(y_production, y_pred_production)
```

### Ejemplo 3: Generación de Datos Sintéticos para Testing

```python
from src.models.data_drift import DataDriftDetector

# Datos de referencia
X_train = pd.read_csv('data/processed/bike_sharing_features_train.csv')

# Inicializar detector
detector = DataDriftDetector(X_train)

# Generar datos con drift simulado
synthetic_drifted = detector.generate_synthetic_drift(
    n_samples=200,
    drift_type="mean_shift",      # 'mean_shift', 'variance_shift', 'distribution_shift'
    drift_magnitude=2.0,          # Magnitud del drift (múltiplo de std)
    features_to_drift=['temp', 'hum', 'windspeed']  # Features a aplicar drift
)

# Verificar que el drift fue aplicado
results = detector.detect_drift(synthetic_drifted)
print(f"Drift simulado detectado: {results['has_drift']}")
print(f"Score: {results['drift_score']:.3f}")

# Usar para validar sistema de detección
```

### Ejemplo 4: Integración con Modelo Entrenado

```python
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.models.data_drift import DataDriftDetector, PerformanceMonitor
from src.config.config_loader import ConfigLoader
from src.config.paths import ProjectPaths

# Configuración
config = ConfigLoader()
paths = ProjectPaths(config)

# 1. Entrenar modelo y obtener baseline
trainer = ModelTrainer(config, paths)
pipeline = trainer.train_model(
    model_type="random_forest",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)

evaluator = ModelEvaluator(config)
y_val_pred = pipeline.predict(X_val)
baseline_metrics = evaluator.evaluate(y_val.values, y_val_pred)

# 2. Configurar monitoreo
detector = DataDriftDetector(X_train)
monitor = PerformanceMonitor(
    baseline_metrics=baseline_metrics,
    performance_threshold=0.2,
    metric_type='mae'
)

# 3. Monitorear en producción
def production_monitoring(X_prod, y_prod):
    # Detectar drift
    drift_results = detector.detect_drift(X_prod)
    
    # Predecir
    y_pred = pipeline.predict(X_prod)
    
    # Monitorear performance
    perf_results = monitor.check_performance(y_prod.values, y_pred)
    
    return {
        'drift': drift_results,
        'performance': perf_results
    }

# Ejecutar monitoreo
results = production_monitoring(X_production, y_production)
```

---

## Tests y Validación

### Ejecutar Tests de Data Drift

```bash
# Todos los tests de drift
make test-drift

# O con pytest directamente
pytest tests/data_drift/ -v

# Tests específicos
pytest tests/data_drift/test_data_drift.py::TestDataDriftDetector::test_mean_shift_drift -v
pytest tests/data_drift/test_data_drift.py::TestPerformanceMonitor -v
```

### Tests Disponibles

**DataDriftDetector:**
- `test_no_drift`: Verifica que datos idénticos no generan drift
- `test_mean_shift_drift`: Detecta drift por cambio de media
- `test_variance_shift_drift`: Detecta drift por cambio de varianza
- `test_categorical_drift`: Detecta drift en features categóricas
- `test_generate_synthetic_drift`: Valida generación de datos sintéticos

**PerformanceMonitor:**
- `test_no_degradation`: Verifica cuando no hay degradación
- `test_mae_degradation`: Detecta degradación en MAE
- `test_r2_degradation`: Detecta degradación en R²
- `test_alert_threshold`: Verifica sistema de alertas

**Integración:**
- `test_drift_detection_with_trained_model`: Integración completa
- `test_drift_with_synthetic_data`: Testing con datos sintéticos
- `test_end_to_end_drift_monitoring`: Flujo completo

### Cobertura de Tests

Los tests cubren:
- ✅ Detección de diferentes tipos de drift
- ✅ Generación de datos sintéticos
- ✅ Monitoreo de performance
- ✅ Integración con modelos entrenados
- ✅ Manejo de edge cases

---

## Mejores Prácticas

### 1. Selección de Features a Monitorear

- **Monitorear features críticas**: Prioriza features con alta importancia
- **Balance**: No monitorees todas las features (ruido), pero incluye las relevantes
- **Features categóricas**: Especifica manualmente si son importantes para el negocio

```python
# Mejor: Monitorear features importantes
important_features = ['temp', 'hum', 'windspeed', 'hr', 'workingday']
detector = DataDriftDetector(
    X_train,
    feature_columns=important_features
)
```

### 2. Configuración de Umbrales

- **threshold (P-value)**: 0.05 es estándar, pero ajusta según tu caso
  - Más estricto (0.01): Menos falsos positivos, más falsos negativos
  - Menos estricto (0.10): Más alertas, pero más sensibilidad
- **psi_threshold**: 0.25 para drift significativo es razonable
- **performance_threshold**: 0.2 (20%) es un buen punto de partida

```python
# Para producción crítica (más estricto)
detector = DataDriftDetector(X_train, threshold=0.01, psi_threshold=0.15)

# Para desarrollo (más permisivo)
detector = DataDriftDetector(X_train, threshold=0.10, psi_threshold=0.30)
```

### 3. Frecuencia de Monitoreo

- **Datos en tiempo real**: Monitoreo continuo o cada hora
- **Datos batch**: Monitoreo después de cada batch
- **Balance costo/beneficio**: Más frecuente = más recursos, más temprana detección

```python
# Ejemplo: Monitoreo diario
def daily_drift_check():
    latest_data = load_latest_production_data()
    results = detector.detect_drift(latest_data)
    log_results(results)
    return results
```

### 4. Acciones ante Drift Detectado

1. **Drift sin degradación**: Monitorear más de cerca
2. **Drift con degradación moderada**: Investigar causas, preparar retrenamiento
3. **Degradación significativa (alert)**: Retrenar modelo inmediatamente

```python
def handle_drift_results(drift_results, perf_results):
    if perf_results['alert']:
        # Acción inmediata
        retrain_model()
        send_notification("Model retraining triggered")
    elif perf_results['has_degradation']:
        # Investigar
        investigate_drift_causes(drift_results)
        schedule_retraining()
    elif drift_results['has_drift']:
        # Monitorear
        increase_monitoring_frequency()
```

### 5. Almacenamiento de Resultados

- Guarda resultados históricos para análisis de tendencias
- Mantén logs de alertas y acciones tomadas
- Usa para análisis post-mortem y mejora continua

```python
import json
from datetime import datetime

def log_monitoring_results(drift_results, perf_results):
    entry = {
        'timestamp': datetime.now().isoformat(),
        'drift_score': drift_results['drift_score'],
        'drift_detected': drift_results['has_drift'],
        'degradation_score': perf_results['degradation_score'],
        'alert_triggered': perf_results['alert']
    }
    
    with open('logs/monitoring_history.jsonl', 'a') as f:
        f.write(json.dumps(entry) + '\n')
```

---

## Solución de Problemas

### Problema: Drift detectado constantemente

**Causa posible**: Umbrales muy estrictos o cambios esperados (estacionalidad)

**Solución**:
```python
# Ajustar umbrales o excluir features estacionales
detector = DataDriftDetector(
    X_train,
    threshold=0.10,  # Menos estricto
    psi_threshold=0.30
)
```

### Problema: No se detecta drift cuando debería

**Causa posible**: Umbrales muy permisivos o drift sutil

**Solución**:
```python
# Usar umbrales más estrictos
detector = DataDriftDetector(
    X_train,
    threshold=0.01,  # Más estricto
    psi_threshold=0.15
)
```

### Problema: Performance degrada sin drift detectado

**Causa posible**: Concept drift (cambio en relación X→y) no detectable por drift de features

**Solución**: Monitorear performance directamente (ya implementado en PerformanceMonitor)

### Problema: PSI muy alto pero p-value no significativo

**Causa posible**: PSI es más sensible a cambios pequeños que tests estadísticos

**Solución**: Confiar en PSI para alertas tempranas, usar p-value para confirmación

```python
# Usar ambos criterios
has_drift = (ks_pvalue < threshold) or (psi >= psi_threshold)
```

---

## Referencias

- **PSI (Population Stability Index)**: 
  - Thresholds: PSI < 0.1 (stable), 0.1-0.25 (moderate change), ≥0.25 (significant change)
- **Kolmogorov-Smirnov Test**: Test no paramétrico para comparar distribuciones
- **Chi-square Test**: Test para comparar distribuciones categóricas

---

## Próximos Pasos

Mejoras futuras posibles:

1. **Concept Drift Detection**: Detectar cambios en relación X→y
2. **Automated Retraining**: Retrenamiento automático cuando se detecta degradación
3. **Dashboard**: Visualización de métricas de drift y performance
4. **Alertas Multi-canal**: Integración con Slack, email, etc.
5. **Drift Explanation**: Explicación de qué features causan el drift

---

## Contacto y Contribuciones

Para preguntas o mejoras, consulta la documentación del proyecto o contacta al equipo.

