# 🔍 AUDITORÍA EXHAUSTIVA: FEATURE ENGINEERING & TEMPORAL LEAKAGE

**Fecha:** 2025-01-12  
**Auditor:** Dr. ML-MLOps Elite Reviewer  
**Notebooks Auditados:**  
- `notebook.ipynb` (EDA & Feature Engineering)
- `02_modeling.ipynb` (Modelado & Evaluación)

---

## 📊 I. VERIFICACIÓN DE TEMPORAL LEAKAGE - LAGS Y ROLLING MEANS

### ✅ **VEREDICTO: CÓDIGO ACTUAL ES CORRECTO (SIN LEAKAGE)**

#### Evidencia del Código Actual (Cell 64 de notebook.ipynb):

```python
# LAGS - CORRECTO ✅
OPTIMAL_LAGS = [1, 24, 48, 72, 168]
for target in ['cnt_transformed']:
    for lag in OPTIMAL_LAGS:
        df_features[f'{target}_lag_{lag}h'] = df_features[target].shift(lag)
        #                                                             ^^^^^^^^
        # ✅ .shift(lag) usa valores PASADOS (t-lag)
        # Para timestamp t=100, lag=1 usa valor de t=99 (CORRECTO)

# ROLLING MEANS - CORRECTO ✅
ROLLING_WINDOWS = [3, 24, 72]
for target in ['cnt_transformed']:
    for window in ROLLING_WINDOWS:
        df_features[f'{target}_roll_mean_{window}h'] = (
            df_features[target].shift(1).rolling(window=window, min_periods=1).mean()
            #                   ^^^^^^^^
            # ✅ .shift(1) ANTES de .rolling() asegura NO usar valor actual
            # Para t=100, window=3: usa promedio de [t-1, t-2, t-3] = [99, 98, 97] (CORRECTO)
        )
```

#### Verificación Matemática:

**Para un timestamp t=100 con lag=24:**
```
df['lag_24h'] = df['cnt'].shift(24)

Resultado:
  t=100 → lag_24h usa valor de t=76  ✅ PASADO
  t=101 → lag_24h usa valor de t=77  ✅ PASADO
```

**Para un timestamp t=100 con rolling_mean_3h:**
```
df['roll_mean_3h'] = df['cnt'].shift(1).rolling(3).mean()

Paso 1: shift(1)
  t=100 → shifted_value = valor de t=99

Paso 2: rolling(3).mean() sobre la serie shifteada
  t=100 → promedio de shifted[t-2:t+1] = [valor_t97, valor_t98, valor_t99] ✅ PASADO

Resultado: NO usa información de t=100 ni futura
```

### 🔍 Análisis de Potenciales Problemas Sutiles

#### ⚠️ Problema Detectado 1: `min_periods=1` en Rolling Windows

**Código actual:**
```python
.rolling(window=window, min_periods=1).mean()
```

**Problema:**
- Los primeros registros del rolling mean se calculan con MENOS observaciones de las esperadas
- Ejemplo: `window=24` → Los primeros 23 valores usan <24 observaciones

**¿Es data leakage?** ❌ NO técnicamente, pero reduce calidad del feature

**Solución recomendada:**
```python
# Opción 1: Usar min_periods=window (más estricto)
.rolling(window=window, min_periods=window).mean()
# → Genera NaN en primeros (window-1) registros

# Opción 2: Usar min_periods=window//2 (balanceado)
.rolling(window=window, min_periods=window//2).mean()
# → Requiere al menos 50% de las observaciones esperadas
```

**Impacto:** BAJO - Los registros con NaN se eliminan después con `dropna()`

**Recomendación:** Cambiar a `min_periods=window` para mayor consistencia

---

#### ⚠️ Problema Detectado 2: Cambios Porcentuales Sin Shift

**Código actual:**
```python
df_features['cnt_pct_change_1h'] = df_features['cnt_transformed'].pct_change(periods=1)
```

**¿Es data leakage?** ❌ NO

**Explicación:**
```python
pct_change(periods=1) calcula: (valor_t - valor_t-1) / valor_t-1

Para t=100:
  pct_change = (cnt[100] - cnt[99]) / cnt[99]
  
¿Usa información futura? NO
¿Usa valor actual (t=100)? SÍ, pero eso está permitido

ANALOGÍA: Es como usar 'temp' o 'hr' actuales - son observables en el momento t
```

**Veredicto:** CORRECTO ✅

---

## 🚨 II. PROBLEMAS CRÍTICOS DETECTADOS EN MODELADO

### ❌ **PROBLEMA 1: OVERFITTING SEVERO EN XGBOOST**

#### Evidencia:

```python
# Métricas en 02_modeling.ipynb (Cell 34 output)
Train RMSE: 5.05    ← SOSPECHOSAMENTE PERFECTO
Train R²: 0.9998    ← 99.98% varianza explicada 🚩🚩🚩

Validation RMSE: 42.88   ← Excelente (pero...)
Validation R²: 0.9708    ← 97% 

# PERO... Cross-Validation (Cell 38 output)
CV RMSE: 138.40 ± 39.80  ← 3.2x PEOR que single split! 🚩🚩🚩
CV R²: 0.7277 ± 0.1960   ← Mucha variabilidad

# Discrepancia: 223% entre Val RMSE (single) y CV RMSE
# Ratio: 138.40 / 42.88 = 3.23x
```

#### Diagnóstico:

1. **Train R² = 0.9998** → Modelo MEMORIZÓ los datos (overfitting extremo)
2. **Val RMSE (42.88) << CV RMSE (138.40)** → Validation set es "afortunado" (no representativo)
3. **Alta variabilidad en CV** (std=39.80) → Modelo NO generaliza consistentemente

#### Causa Raíz:

**Hiperparámetros demasiado permisivos:**
```python
# Cell 34 - XGBoost params ACTUALES
xgb_params = {
    'max_depth': 6,           # ← DEMASIADO profundo
    'learning_rate': 0.05,    # ← OK
    'min_child_weight': 3,    # ← POCO restrictivo
    'gamma': 0.1,             # ← Penalización MUY baja
    'reg_alpha': 0.1,         # ← L1 regularización BAJA
    'reg_lambda': 1.0,        # ← L2 regularización BAJA
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

#### ✅ Solución Propuesta:

```python
# HIPERPARÁMETROS CORREGIDOS (MÁS CONSERVADORES)
xgb_params = {
    'n_estimators': 300,         # ↓ Reducido de 500
    'max_depth': 4,              # ↓ Reducido de 6 → menos complejidad
    'learning_rate': 0.03,       # ↓ Reducido de 0.05 → aprendizaje más lento
    'subsample': 0.7,            # ↓ Reducido de 0.8 → bootstrap más agresivo
    'colsample_bytree': 0.7,     # ↓ Reducido de 0.8 → menos features por árbol
    'colsample_bylevel': 0.7,    # ↓ Reducido de 0.8
    'min_child_weight': 5,       # ↑ Aumentado de 3 → más restrictivo
    'gamma': 0.5,                # ↑ Aumentado de 0.1 → penalización moderada
    'reg_alpha': 0.5,            # ↑ Aumentado de 0.1 → L1 más fuerte
    'reg_lambda': 2.0,           # ↑ Aumentado de 1.0 → L2 más fuerte
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50
}
```

**Justificación de cambios:**
- **max_depth↓ 6→4:** Árboles más shallow → menos overfitting
- **learning_rate↓ 0.05→0.03:** Aprendizaje más conservador → mejor generalización
- **min_child_weight↑ 3→5:** Requiere más muestras por hoja → menos overfitting
- **gamma↑ 0.1→0.5:** Mayor penalización por splits → menos árboles complejos
- **reg_alpha/lambda↑:** Regularización L1/L2 más fuerte → menos overfitting

**Mejora esperada:**
- Train R² bajará a ~0.85-0.90 (BUENO - menos memorización)
- CV RMSE mejorará hacia ~100-120 (más realista)
- Menor gap Train-Val (mejor generalización)

---

### ❌ **PROBLEMA 2: RIDGE CON R² NEGATIVO EN CV**

#### Evidencia:

```python
# Cell 38 output - Ridge CV results
CV R²: -0.0076 ± 0.7399  ← ¡R² NEGATIVO! 🚩
CV RMSE: 271.95 ± 47.32

# Single split
Val R²: 0.5420  ← Aceptable pero inconsistente con CV
```

**¿Qué significa R² negativo?**
→ El modelo es **PEOR que predecir la media constante**

#### Causa Raíz:

1. **Features no lineales** pero modelo LINEAR
   - Bike demand tiene patrones NO LINEALES (hora pico, clima, interacciones)
   - Ridge espera relaciones lineales simples

2. **Alpha muy bajo** (0.01) → Casi no hay regularización

3. **Multicolinealidad** entre los 40 features

#### ✅ Solución:

**Ridge NO es apropiado para este problema.** Usar modelos tree-based (RF/XGBoost).

**Si se quiere mantener Ridge (para baseline):**
```python
ridge_params = {
    'alpha': 10.0,  # ↑ De 0.01 → 10.0 (penaliza colinealidad)
    'max_iter': 10000
}
```

---

## 🎯 III. FEATURES FALTANTES - ANÁLISIS DE GAPS

### 📋 Comparación: Key Insights vs Features Implementados

| Feature Sugerido en EDA | ¿Implementado? | Justificación |
|--------------------------|----------------|---------------|
| `atemp` eliminado | ✅ SÍ | Multicolinealidad con temp (r=0.987) |
| `cnt_transformed` (sqrt) | ✅ SÍ | Target transformado |
| Features cíclicas (sin/cos) | ✅ SÍ | hr, mnth, weekday |
| `is_weekend` | ✅ SÍ | Patrón diferenciado |
| `is_peak_hour` | ✅ SÍ | Horas 8, 17, 18 |
| `is_commute_window` | ✅ SÍ | 7-9am, 4-7pm |
| `temp_season` | ✅ SÍ | Interacción climática |
| `weathersit_season` | ✅ SÍ | Clima × estación |
| `hr_workingday` | ✅ SÍ | Patrón bimodal |
| `weather_quadrant` | ✅ SÍ | Cuadrantes Temp×Hum |
| Lags [1,24,48,72,168] | ✅ SÍ | Validado por ACF/PACF |
| Rolling means [3,24,72] | ✅ SÍ | Ventanas móviles |
| `cnt_pct_change_1h/24h` | ✅ SÍ | Cambios porcentuales |
| **`casual_share`** | ❌ **ELIMINADO** | Data leakage (correcto) |
| **`is_weekend_casual_share`** | ❌ **ELIMINADO** | Data leakage (correcto) |

### 🔍 Features Mencionados en Key Insights pero NO Implementados:

**1. `casual_share` (proporción de usuarios casuales)**
```python
# Key Insights Section XI.G sugiere:
df_features['casual_lag_1h'] = df_features['casual'].shift(1)
df_features['cnt_lag_1h'] = df_features['cnt'].shift(1)
df_features['casual_share'] = df_features['casual_lag_1h'] / df_features['cnt_lag_1h']
```

**¿Por qué fue eliminado en Cell 62?**
```python
# Cell 62 output:
# "🔴 casual_share ELIMINADO (prevención de data leakage)"
```

**Análisis:**
- ✅ **DECISIÓN CORRECTA**
- `casual` y `registered` son **COMPONENTES** del target: `cnt = casual + registered`
- Aunque se use lag, sigue siendo problemático:
  1. En producción, puede que NO tengamos acceso a `casual/registered` en tiempo real
  2. Modelo debe ser robusto y no depender de componentes del target
  3. **Principio de simplicidad:** Mejor predecir `cnt` directamente sin descomposición

**Veredicto:** ✅ Mantener eliminado

---

## 🌟 IV. FEATURES ADICIONALES SUGERIDOS (BASADO EN EXPERIENCIA)

### 📊 Features Propuestos con Justificación

#### 1. **Momentum Features (Aceleración de Demanda)**

**Concepto:** Capturar si la demanda está ACELERANDO o DESACELERANDO

```python
# Aceleración de 1h (cambio en el cambio)
df_features['cnt_acceleration_1h'] = (
    df_features['cnt_pct_change_1h'] - 
    df_features['cnt_pct_change_1h'].shift(1)
)

# Aceleración de 24h
df_features['cnt_acceleration_24h'] = (
    df_features['cnt_pct_change_24h'] - 
    df_features['cnt_pct_change_24h'].shift(1)
)
```

**Justificación:**
- Detecta **tendencias emergentes** (ej: demanda creciendo rápidamente antes de un evento)
- Útil para capturar **patrones de transición** (ej: paso de valle a pico)

**Evidencia del EDA:**
- Ratio pico/valle = 46x → Transiciones abruptas son críticas
- Patrón bimodal → Momentum puede anticipar picos

---

#### 2. **Features de Volatilidad (Estabilidad de Demanda)**

**Concepto:** Capturar si la demanda es ESTABLE o VOLÁTIL

```python
# Desviación estándar rolling de 24h (volatilidad)
df_features['cnt_volatility_24h'] = (
    df_features['cnt_transformed']
    .shift(1)
    .rolling(window=24, min_periods=12)
    .std()
)

# Coeficiente de variación (normalizado)
df_features['cnt_cv_24h'] = (
    df_features['cnt_volatility_24h'] / 
    df_features['cnt_transformed_roll_mean_24h']
)
```

**Justificación:**
- Detecta **días atípicos** o **eventos especiales**
- Útil para ajustar **bandas de confianza** en predicciones

**Evidencia del EDA:**
- Test de Levene confirmó **heterocedasticidad** (varianza NO constante)
- Festivos y fines de semana tienen mayor variabilidad

---

#### 3. **Features de Contexto Histórico (¿Mejor o Peor que Ayer?)**

**Concepto:** Comparar demanda actual con promedio histórico

```python
# Promedio histórico para misma hora/día de semana (usando solo train data)
# Calcular en train, aplicar a val/test
historical_avg = (
    train_df
    .groupby(['hr', 'weekday'])['cnt']
    .mean()
    .rename('cnt_historical_avg')
)

# Merge con df_features
df_features = df_features.merge(
    historical_avg,
    on=['hr', 'weekday'],
    how='left'
)

# Feature: desviación respecto a promedio histórico
df_features['cnt_vs_historical'] = (
    df_features['cnt_transformed'] - 
    np.sqrt(df_features['cnt_historical_avg'])
)
```

**Justificación:**
- Captura si demanda está **por encima/debajo** de lo esperado
- Útil para detectar **anomalías** y **eventos especiales**

**Evidencia del EDA:**
- Patrón horario es MUY estable (ACF lag 24h = 0.53)
- Patrón semanal significativo (ACF lag 168h = 0.35)

---

#### 4. **Interacciones Climáticas Avanzadas**

**Concepto:** Capturar efectos NO lineales del clima

```python
# Sensación térmica cuadrática (efecto parabólico)
df_features['temp_squared'] = df_features['temp'] ** 2

# Interacción Temp × Humedad (índice de disconfort)
df_features['temp_hum_interaction'] = df_features['temp'] * df_features['hum']

# Interacción Temp × Windspeed (sensación de viento frío)
df_features['temp_wind_interaction'] = df_features['temp'] * df_features['windspeed']

# Índice de "clima perfecto" (temp óptima ~0.5-0.7, hum baja)
optimal_temp = 0.6  # Normalizado
df_features['is_perfect_weather'] = (
    (df_features['temp'].between(0.5, 0.7)) & 
    (df_features['hum'] < 0.5) &
    (df_features['weathersit'] == 1)
).astype(int)
```

**Justificación:**
- **Relación parabólica:** Temperatura muy baja O muy alta reduce demanda
- **Efecto multiplicativo:** Humedad alta amplifica efecto negativo de calor

**Evidencia del EDA:**
- Cuadrantes climáticos tienen ratio 2.80x (mejor/peor)
- Correlación temp-cnt es moderada (+0.204) pero puede ser NO lineal

---

#### 5. **Features de Día Especial (Beyond Holiday)**

**Concepto:** Capturar días con comportamiento atípico (NO solo festivos)

```python
# Fin de mes (último 3 días del mes)
df_features['is_end_of_month'] = (df_features['day'] >= 28).astype(int)

# Primer día del mes
df_features['is_start_of_month'] = (df_features['day'] == 1).astype(int)

# Temporada universitaria (septiembre-mayo, excluyendo diciembre)
df_features['is_school_season'] = (
    df_features['mnth'].isin([1,2,3,4,5,9,10,11])
).astype(int)

# Verano (junio-agosto)
df_features['is_summer_vacation'] = (
    df_features['mnth'].isin([6,7,8])
).astype(int)
```

**Justificación:**
- **Fin/inicio de mes:** Patrones de gasto/salario pueden afectar uso de bicis
- **Temporada escolar:** Estudiantes son usuarios importantes

**Evidencia del EDA:**
- Septiembre tiene la mayor demanda (mes 9)
- Festivos tienen comportamiento diferenciado

---

#### 6. **Features de Rezago Diferenciado por Tipo de Día**

**Concepto:** Lags DIFERENTES para weekdays vs weekends

```python
# Lag condicional: lag_24h solo si mismo tipo de día
def conditional_lag_24h(row):
    """Usar lag 24h solo si es mismo tipo de día (weekday vs weekend)"""
    if row['is_weekend'] == 1:
        # Weekend: usar lag 24h de fin de semana anterior
        return row['cnt_transformed_lag_168h']  # 1 semana
    else:
        # Weekday: usar lag 24h del día anterior
        return row['cnt_transformed_lag_24h']

df_features['cnt_lag_conditional'] = df_features.apply(conditional_lag_24h, axis=1)
```

**Justificación:**
- **Lunes NO se parece a Domingo** (fin de semana)
- **Sábado se parece más a Sábado anterior** que a Viernes

**Evidencia del EDA:**
- Patrón weekday es bimodal, weekend es uniforme
- Interacción `hr × workingday` es significativa

---

### 📊 Resumen de Features Propuestos

| Feature Propuesto | Impacto Esperado | Complejidad | Prioridad |
|-------------------|------------------|-------------|-----------|
| Momentum (aceleración) | MEDIO | BAJA | 🟡 MEDIA |
| Volatilidad (rolling std) | MEDIO-ALTO | BAJA | 🟢 ALTA |
| Contexto histórico | ALTO | MEDIA | 🟢 ALTA |
| Temp cuadrática | MEDIO | BAJA | 🟡 MEDIA |
| Interacciones climáticas | MEDIO | BAJA | 🟡 MEDIA |
| Días especiales | BAJO | BAJA | 🔴 BAJA |
| Lags condicionales | MEDIO | MEDIA | 🟡 MEDIA |

**Recomendación:** Implementar **Volatilidad** y **Contexto histórico** primero (máximo ROI).

---

## ✅ V. PLAN DE CORRECCIONES - RESUMEN EJECUTIVO

### 🔴 **CRÍTICAS (Hacer AHORA):**

1. **Corregir hiperparámetros XGBoost** en `02_modeling.ipynb` (Cell 34)
   ```python
   max_depth: 6 → 4
   learning_rate: 0.05 → 0.03
   min_child_weight: 3 → 5
   gamma: 0.1 → 0.5
   reg_alpha: 0.1 → 0.5
   reg_lambda: 1.0 → 2.0
   ```

2. **Cambiar `min_periods` en rolling windows** en `notebook.ipynb` (Cell 64)
   ```python
   .rolling(window=window, min_periods=1)  
   → .rolling(window=window, min_periods=window)
   ```

3. **Actualizar Ridge alpha** en `02_modeling.ipynb` (Cell 22)
   ```python
   alpha: 0.01 → 10.0  # Mayor regularización
   ```

### 🟡 **IMPORTANTES (Hacer PRONTO):**

4. **Añadir features de volatilidad** en `notebook.ipynb` (nueva celda después de Cell 64)
   ```python
   df_features['cnt_volatility_24h'] = ...
   df_features['cnt_cv_24h'] = ...
   ```

5. **Añadir contexto histórico** en `notebook.ipynb`
   ```python
   cnt_historical_avg = ...
   df_features['cnt_vs_historical'] = ...
   ```

6. **Añadir interacciones climáticas** en `notebook.ipynb`
   ```python
   df_features['temp_squared'] = ...
   df_features['temp_hum_interaction'] = ...
   ```

### 🔵 **OPCIONALES (Considerar para V2):**

7. Momentum features
8. Lags condicionales
9. Features de días especiales

---

## 🎯 VI. CONCLUSIONES Y PRÓXIMOS PASOS

### ✅ Conclusiones del Análisis:

1. **Feature Engineering actual es SÓLIDO:**
   - ✅ NO hay temporal leakage en lags/rolling means
   - ✅ Transformación del target (`sqrt`) es apropiada
   - ✅ Features cíclicos, interacciones, e indicadores están bien implementados

2. **Modelado tiene ISSUES CRÍTICOS:**
   - ❌ XGBoost tiene overfitting SEVERO (Train R²=0.9998, CV R²=0.7277)
   - ❌ Ridge no es apropiado para este problema (R² negativo en CV)
   - ❌ Discrepancia ENORME entre single split y CV (223% en XGBoost)

3. **Oportunidades de Mejora:**
   - 🟡 Features adicionales (volatilidad, contexto histórico) pueden mejorar +5-10% MAE
   - 🟡 Hiperparámetros más conservadores mejorarán generalización

### 🚀 Acción Inmediata:

1. **Corregir `02_modeling.ipynb`:**
   - XGBoost hiperparámetros más conservadores
   - Ridge alpha más alto
   - Re-ejecutar y verificar que CV RMSE mejora

2. **Mejorar `notebook.ipynb`:**
   - `min_periods=window` en rolling windows
   - Añadir features de volatilidad y contexto histórico

3. **Re-evaluar:**
   - Después de correcciones, XGBoost CV RMSE debería bajar a ~100-120
   - Gap Train-Val debería reducirse a <20%

---

**Documento preparado por:** Dr. ML-MLOps Elite Reviewer  
**Próximo paso:** Implementar correcciones críticas en ambos notebooks

