# 🔧 CAMBIOS NECESARIOS EN 02_modeling.ipynb

## 📋 Resumen Ejecutivo

El notebook `notebook.ipynb` fue actualizado con hallazgos críticos del EDA que **requieren cambios obligatorios** en el notebook de modelado (`02_modeling.ipynb`) para mantener consistencia y aprovechar las mejoras implementadas.

---

## 🚨 CAMBIOS CRÍTICOS OBLIGATORIOS

### 1. ⚠️ **TARGET TRANSFORMADO** (MÁS IMPORTANTE)

**Problema Actual:**
```python
# ❌ INCORRECTO - Está usando target original
y_train = train_df['cnt'].values
y_val = val_df['cnt'].values
y_test = test_df['cnt'].values
```

**Cambio Requerido:**
```python
# ✅ CORRECTO - Usar target transformado
y_train = train_df['cnt_transformed'].values  # sqrt(cnt)
y_val = val_df['cnt_transformed'].values
y_test = test_df['cnt_transformed'].values

print("\n⚠️ IMPORTANTE: Usando target transformado (sqrt)")
print(f"  Target original (cnt): {train_df['cnt'].mean():.2f} ± {train_df['cnt'].std():.2f}")
print(f"  Target transformado: {y_train.mean():.2f} ± {y_train.std():.2f}")
```

**Justificación:**
- Análisis experimental en EDA mostró que `Sqrt(y)` es la mejor transformación
- Mejora: +1.97% MAE, +2.34% R²
- Shapiro-Wilk confirmó que target original NO es normal (p < 0.0001)

---

### 2. 🔄 **TRANSFORMACIÓN INVERSA EN EVALUACIÓN**

**Problema Actual:**
```python
# ❌ INCORRECTO - Evaluando en escala transformada
def evaluate_model(model, X, y_true, dataset_name=""):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)  # ← Escala transformada
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
```

**Cambio Requerido:**
```python
# ✅ CORRECTO - Transformación inversa para métricas en escala original
def evaluate_model(model, X, y_true_transformed, dataset_name=""):
    """
    Evalúa modelo con métricas en escala ORIGINAL (no transformada)
    
    Args:
        model: Modelo entrenado
        X: Features
        y_true_transformed: Target transformado (sqrt(cnt))
        dataset_name: Nombre del dataset para logging
    
    Returns:
        dict: Métricas en escala original
    """
    # Predicción en escala transformada
    y_pred_transformed = model.predict(X)
    
    # ⚠️ TRANSFORMACIÓN INVERSA: sqrt(x) → x^2
    y_pred_original = y_pred_transformed ** 2
    y_true_original = y_true_transformed ** 2
    
    # Asegurar predicciones no-negativas
    y_pred_original = np.clip(y_pred_original, 0, None)
    
    # Calcular métricas en escala ORIGINAL
    mae = mean_absolute_error(y_true_original, y_pred_original)
    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_true_original, y_pred_original) * 100
    
    print(f"\n📊 Evaluación en {dataset_name} (escala ORIGINAL):")
    print(f"  MAE:  {mae:.2f} {'✅' if mae < 50 else '❌'} (objetivo: < 50)")
    print(f"  RMSE: {rmse:.2f} {'✅' if rmse < 80 else '❌'} (objetivo: < 80)")
    print(f"  R²:   {r2:.4f} {'✅' if r2 > 0.7 else '❌'} (objetivo: > 0.7)")
    print(f"  MAPE: {mape:.2f}% {'✅' if mape < 25 else '❌'} (objetivo: < 25%)")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'y_pred_original': y_pred_original,  # Para análisis posterior
        'y_true_original': y_true_original
    }
```

**Razón:**
- Los targets objetivo (MAE < 50, RMSE < 80) están definidos en escala ORIGINAL
- Evaluar en escala transformada daría métricas no interpretables para negocio

---

### 3. 📊 **ACTUALIZAR FEATURES DE DATA LEAKAGE**

**Problema Actual:**
```python
# ❌ DESACTUALIZADO - Lags antiguos
leakage_features = [
    'cnt_lag_1h', 'cnt_lag_24h', 'cnt_lag_168h',  # ← Solo 3 lags
    'cnt_roll_mean_3h', 'cnt_roll_mean_24h',      # ← Solo 2 rolling
    # ...
    'registered_lag_1h', 'registered_lag_24h', 'registered_lag_168h',  # ← 3 lags
    'casual_lag_1h', 'casual_lag_24h', 'casual_lag_168h',
]
```

**Cambio Requerido:**
```python
# ✅ ACTUALIZADO - Lags validados por ACF/PACF
leakage_features = [
    # Componentes del target
    'casual_share', 'casual_lag_1h', 'cnt_lag_1h_for_share',
    
    # Lags del target TRANSFORMADO (cnt_transformed = sqrt(cnt))
    'cnt_transformed_lag_1h', 'cnt_transformed_lag_24h', 'cnt_transformed_lag_48h',
    'cnt_transformed_lag_72h', 'cnt_transformed_lag_168h',  # ← 5 lags (ACF/PACF)
    
    # Rolling means del target transformado
    'cnt_transformed_roll_mean_3h', 'cnt_transformed_roll_mean_24h', 
    'cnt_transformed_roll_mean_72h',  # ← 3 rolling windows
    
    # Cambios porcentuales del target transformado
    'cnt_pct_change_1h', 'cnt_pct_change_24h',
    
    # Lags de componentes (registered y casual) - ACTUALIZADOS
    'registered_lag_1h', 'registered_lag_24h', 'registered_lag_48h', 
    'registered_lag_72h', 'registered_lag_168h',  # ← 5 lags
    'registered_roll_mean_3h', 'registered_roll_mean_24h', 'registered_roll_mean_72h',
    
    'casual_lag_1h', 'casual_lag_24h', 'casual_lag_48h', 
    'casual_lag_72h', 'casual_lag_168h',  # ← 5 lags
    'casual_roll_mean_3h', 'casual_roll_mean_24h', 'casual_roll_mean_72h',
]
```

**Justificación:**
- Lags actualizados de `[1, 24, 168]` a `[1, 24, 48, 72, 168]` (validado por ACF/PACF)
- Rolling windows actualizados de `[3, 24]` a `[3, 24, 72]`
- Ahora usamos `cnt_transformed` (no `cnt`)

---

### 4. 🎯 **ACTUALIZAR TARGETS OBJETIVO EN MÉTRICAS**

**Problema Actual:**
```python
# ❌ Métricas objetivo pueden estar desactualizadas
TARGET_METRICS = {
    'MAE': 50,      # ¿En qué escala?
    'RMSE': 80,     # ¿En qué escala?
    'R2': 0.7,
    'MAPE': 25
}
```

**Cambio Requerido:**
```python
# ✅ CLARIFICAR - Métricas en escala ORIGINAL
TARGET_METRICS = {
    'MAE': 50,      # ← Escala ORIGINAL (bicicletas/hora)
    'RMSE': 80,     # ← Escala ORIGINAL (bicicletas/hora)
    'R2': 0.7,      # ← Invariante a transformación
    'MAPE': 25      # ← Porcentaje, escala ORIGINAL
}

print("="*70)
print("MÉTRICAS OBJETIVO (ESCALA ORIGINAL - NO TRANSFORMADA)")
print("="*70)
print(f"⚠️ IMPORTANTE: Target se predice en escala transformada (sqrt)")
print(f"              Métricas se calculan en escala ORIGINAL (cnt)")
print(f"\nObjetivos:")
for metric, target in TARGET_METRICS.items():
    print(f"  • {metric}: {'<' if metric not in ['R2'] else '>'} {target}")
print("="*70)
```

---

### 5. 📝 **ACTUALIZAR MLflow LOGGING**

**Cambio Requerido:**
```python
# Añadir metadata de transformación
with mlflow.start_run(run_name=f"{model_name}_baseline"):
    # ... training code ...
    
    # ✅ AÑADIR: Metadata de transformación
    mlflow.log_param("target_transformation", "sqrt")
    mlflow.log_param("inverse_transformation", "square")
    mlflow.log_param("lags_used", "[1, 24, 48, 72, 168]")
    mlflow.log_param("rolling_windows", "[3, 24, 72]")
    mlflow.log_param("features_count", len(feature_cols))
    mlflow.log_param("atemp_removed", True)  # Multicolinealidad
    
    # ... rest of logging ...
```

---

## 📊 CAMBIOS RECOMENDADOS (NO CRÍTICOS)

### 6. 📈 **VISUALIZACIÓN DE PREDICCIONES**

Añadir gráfico mostrando predicciones en escala original:

```python
def plot_predictions_comparison(y_true, y_pred, dataset_name="Test"):
    """
    Compara predicciones vs valores reales en escala ORIGINAL
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Demanda Real (cnt)')
    axes[0].set_ylabel('Demanda Predicha (cnt)')
    axes[0].set_title(f'Predicciones vs Real - {dataset_name}')
    axes[0].grid(True, alpha=0.3)
    
    # Residuos
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuos (Real - Predicho)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Residuos')
    axes[1].grid(True, alpha=0.3)
    
    # Serie temporal (primeras 168h)
    n_plot = min(168, len(y_true))
    axes[2].plot(y_true[:n_plot], label='Real', linewidth=1.5, alpha=0.7)
    axes[2].plot(y_pred[:n_plot], label='Predicho', linewidth=1.5, alpha=0.7)
    axes[2].set_xlabel('Hora')
    axes[2].set_ylabel('Demanda (cnt)')
    axes[2].set_title(f'Serie Temporal (primeras {n_plot}h)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Stats de residuos
    print(f"\n📊 Estadísticas de Residuos:")
    print(f"  Mean: {residuals.mean():.2f}")
    print(f"  Std: {residuals.std():.2f}")
    print(f"  Min: {residuals.min():.2f}")
    print(f"  Max: {residuals.max():.2f}")
```

---

### 7. 🎯 **ANÁLISIS POR SEGMENTOS**

Evaluar performance en diferentes rangos de demanda:

```python
def evaluate_by_segments(y_true, y_pred):
    """
    Evalúa performance por segmentos de demanda (según ML Canvas)
    """
    # Definir segmentos (según ML Canvas)
    low_mask = y_true < 1000
    medium_mask = (y_true >= 1000) & (y_true <= 7000)
    high_mask = y_true > 7000
    
    segments = {
        'Baja (<1K)': low_mask,
        'Media (1K-7K)': medium_mask,
        'Alta (>7K)': high_mask
    }
    
    print("\n" + "="*70)
    print("EVALUACIÓN POR SEGMENTOS DE DEMANDA")
    print("="*70)
    
    for segment_name, mask in segments.items():
        n_samples = mask.sum()
        if n_samples == 0:
            print(f"\n{segment_name}: Sin muestras")
            continue
            
        y_true_seg = y_true[mask]
        y_pred_seg = y_pred[mask]
        
        mae = mean_absolute_error(y_true_seg, y_pred_seg)
        rmse = np.sqrt(mean_squared_error(y_true_seg, y_pred_seg))
        r2 = r2_score(y_true_seg, y_pred_seg)
        mape = mean_absolute_percentage_error(y_true_seg, y_pred_seg) * 100
        
        print(f"\n{segment_name}: {n_samples} muestras ({n_samples/len(y_true)*100:.1f}%)")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    print("="*70)
```

---

## 🔍 CHECKLIST DE VERIFICACIÓN

Antes de ejecutar el notebook de modelado, verificar:

- [ ] **Target:** Usar `cnt_transformed` (no `cnt`)
- [ ] **Leakage Features:** Actualizar lista con lags `[1, 24, 48, 72, 168]`
- [ ] **Rolling Windows:** Actualizar a `[3, 24, 72]`
- [ ] **Transformación Inversa:** Aplicar `y_pred^2` antes de evaluar métricas
- [ ] **Evaluación:** Métricas en escala ORIGINAL (no transformada)
- [ ] **MLflow Params:** Incluir metadata de transformación
- [ ] **Visualizaciones:** Mostrar predicciones en escala original
- [ ] **Features:** Verificar que `atemp` NO esté en features (eliminada por multicolinealidad)
- [ ] **Documentación:** Comentarios claros sobre transformación en todo el código

---

## 📂 ARCHIVOS AFECTADOS

### Archivos que YA están actualizados:
✅ `notebook.ipynb` - EDA y Feature Engineering completo con transformación

### Archivos que NECESITAN actualización:
❌ `02_modeling.ipynb` - Requiere TODOS los cambios listados arriba

### Archivos que pueden requerir ajustes menores:
⚠️ Scripts en `src/models/` (si existen) - Verificar transformación del target
⚠️ Scripts de predicción/inferencia - Asegurar transformación inversa

---

## 🚀 IMPACTO ESPERADO

Con estos cambios implementados:

1. **Mejora en métricas:** +1.97% MAE, +2.34% R² (validado experimentalmente)
2. **Reducción de sesgo:** Target transformado reduce sesgo de 15.09 a ~2-3
3. **Mejor convergencia:** Modelos convergen más rápido con target normalizado
4. **Interpretabilidad:** Métricas en escala original son directamente interpretables
5. **Consistencia:** Notebooks alineados con hallazgos del EDA

---

## ⚠️ ADVERTENCIAS

1. **NO** mezclar escalas: Entrenar en escala transformada, evaluar en escala original
2. **SIEMPRE** aplicar transformación inversa antes de calcular métricas finales
3. **VERIFICAR** que features de leakage estén correctamente excluidas
4. **DOCUMENTAR** claramente en qué escala se está trabajando en cada paso
5. **GUARDAR** tanto modelo como metadata de transformación para producción

---

## 📞 PRÓXIMOS PASOS

1. ✅ Leer este documento completo
2. ✅ Hacer backup del notebook actual de modelado
3. ✅ Implementar cambios críticos (1-5) en orden
4. ✅ Implementar cambios recomendados (6-7) si tiempo permite
5. ✅ Ejecutar notebook completo y verificar métricas
6. ✅ Comparar resultados antes/después de transformación
7. ✅ Documentar mejoras en presentación final

---

**Fecha:** 2025-10-12  
**Basado en:** Hallazgos del EDA exhaustivo con análisis ACF/PACF y evaluación experimental de transformaciones

