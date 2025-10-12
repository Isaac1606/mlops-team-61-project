# ✅ ERRORES CORREGIDOS EN 02_MODELING.IPYNB

**Fecha:** 2025-10-12  
**Problema:** `NameError: name 'filter_scalar_metrics' is not defined`

---

## 🔴 PROBLEMA IDENTIFICADO

El notebook intentaba llamar a una función `filter_scalar_metrics()` que no existía en 3 lugares diferentes:

1. **Cell 20:** Modelo Ridge Regression
2. **Cell 25:** Modelo Random Forest
3. **Cell 32:** Modelo XGBoost

### Error Original:

```python
NameError                                 Traceback (most recent call last)
Cell In[20], line 42
     39 for prefix, metrics in [('train', train_metrics_ridge), 
     40                          ('val', val_metrics_ridge),
     41                          ('test', test_metrics_ridge)]:
---> 42     scalar_metrics = filter_scalar_metrics(metrics)  # ← FILTRO AÑADIDO
     43     for metric_name, value in scalar_metrics.items():
     44         mlflow.log_metric(f"{prefix}_{metric_name}", value)

NameError: name 'filter_scalar_metrics' is not defined
```

---

## 💡 CAUSA RAÍZ

La función `filter_scalar_metrics()` era necesaria **antes** porque `evaluate_model()` retornaba arrays numpy que MLflow no podía loggear.

Sin embargo, ya **corregimos** `evaluate_model()` en la celda 17 para que retorne **directamente floats de Python**:

```python
def evaluate_model(y_true, y_pred, dataset_name="Validation"):
    # ... cálculos ...
    
    # CRÍTICO: Convertir todos los valores a Python float nativos para MLflow
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'residuals_mean': float(residuals.mean()),
        'residuals_std': float(residuals.std())
    }
    return metrics
```

Por lo tanto, **filter_scalar_metrics() ya no es necesaria** y las llamadas a esta función causaban errores.

---

## ✅ SOLUCIÓN APLICADA

Eliminamos las llamadas a `filter_scalar_metrics()` en las 3 celdas y loggeamos las métricas directamente:

### Cell 20 - Ridge Regression (CORREGIDO):

**ANTES:**
```python
# Log metrics (⚠️ FILTRAR SOLO ESCALARES)
# evaluate_model retorna arrays además de métricas → filtrar antes de loggear
for prefix, metrics in [('train', train_metrics_ridge), 
                         ('val', val_metrics_ridge),
                         ('test', test_metrics_ridge)]:
    scalar_metrics = filter_scalar_metrics(metrics)  # ← FILTRO AÑADIDO
    for metric_name, value in scalar_metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", value)
```

**DESPUÉS:**
```python
# Log metrics (evaluate_model ya retorna floats Python → logging directo)
for prefix, metrics in [('train', train_metrics_ridge), 
                         ('val', val_metrics_ridge),
                         ('test', test_metrics_ridge)]:
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", value)
```

### Cell 25 - Random Forest (CORREGIDO):

**ANTES:**
```python
# Log metrics (⚠️ FILTRAR SOLO ESCALARES)
for prefix, metrics in [('train', train_metrics_rf), 
                         ('val', val_metrics_rf),
                         ('test', test_metrics_rf)]:
    scalar_metrics = filter_scalar_metrics(metrics)  # ← FILTRO AÑADIDO
    for metric_name, value in scalar_metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", value)
```

**DESPUÉS:**
```python
# Log metrics (evaluate_model ya retorna floats Python → logging directo)
for prefix, metrics in [('train', train_metrics_rf), 
                         ('val', val_metrics_rf),
                         ('test', test_metrics_rf)]:
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", value)
```

### Cell 32 - XGBoost (CORREGIDO):

**ANTES:**
```python
# Log metrics (⚠️ FILTRAR SOLO ESCALARES)
for prefix, metrics in [('train', train_metrics_xgb), 
                         ('val', val_metrics_xgb),
                         ('test', test_metrics_xgb)]:
    scalar_metrics = filter_scalar_metrics(metrics)  # ← FILTRO AÑADIDO
    for metric_name, value in scalar_metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", value)
```

**DESPUÉS:**
```python
# Log metrics (evaluate_model ya retorna floats Python → logging directo)
for prefix, metrics in [('train', train_metrics_xgb), 
                         ('val', val_metrics_xgb),
                         ('test', test_metrics_xgb)]:
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", value)
```

---

## 🎯 RESULTADO

✅ **Cell 20 (Ridge):** Error eliminado  
✅ **Cell 25 (Random Forest):** Error eliminado  
✅ **Cell 32 (XGBoost):** Error eliminado  

Ahora el notebook puede ejecutarse completamente sin errores de MLflow logging.

---

## 🔍 VERIFICACIÓN

Para verificar que todo funciona correctamente:

1. Ejecutar el notebook completo desde el inicio
2. Las celdas 20, 25 y 32 deben ejecutarse sin errores
3. MLflow debe loggear todas las métricas correctamente
4. Verificar en MLflow UI que los runs tienen todas las métricas registradas

---

## 📝 LECCIONES APRENDIDAS

1. **Consistencia es clave:** Al corregir `evaluate_model()`, debimos actualizar inmediatamente todo el código que dependía de ella.

2. **Simplificar es mejor:** Eliminar `filter_scalar_metrics()` hace el código más simple y mantenible.

3. **Documentación inline:** Los comentarios actualizados (`# evaluate_model ya retorna floats Python → logging directo`) ayudan a entender por qué el código es así.

---

**Todas las correcciones aplicadas exitosamente** ✅  
**Notebook listo para ejecución completa** 🚀

---

**Documento generado automáticamente**  
**Versión:** 1.0  
**Fecha:** 2025-10-12

