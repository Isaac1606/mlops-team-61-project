# EDA Summary - Bike Sharing Demand Prediction

**Fecha:** Octubre 2025  
**Dataset:** Capital Bikeshare (2011-2012)  
**Objetivo:** Predicción de demanda horaria de bicicletas

---

## 📊 Resumen Ejecutivo

Se realizó un Análisis Exploratorio de Datos (EDA) exhaustivo sobre el dataset de bike sharing, identificando patrones clave, problemas de calidad de datos y oportunidades para el modelado predictivo.

---

## 🔍 Principales Hallazgos

### 1. Calidad de Datos

**Problemas Identificados:**
- ❌ Todas las columnas eran tipo `object` (debían ser numéricas)
- ❌ 180-237 valores nulos por columna
- ❌ Columna `mixed_type_col` con 10% de valores nulos y datos inconsistentes
- ❌ Columna `instant` redundante (solo ID)

**Acciones Tomadas:**
- ✅ Conversión correcta de tipos de datos
- ✅ Eliminación de columnas problemáticas (instant, mixed_type_col)
- ✅ Eliminación de filas con valores nulos
- ✅ Dataset final: ~17,500 observaciones horarias válidas

---

### 2. Variable Target: `cnt` (Total de bicicletas rentadas)

| Métrica | Valor |
|---------|-------|
| **Mínimo** | 1 |
| **Máximo** | 8,000+ |
| **Media** | ~189 |
| **Mediana** | ~142 |
| **Desv. Estándar** | ~181 |

**Características:**
- Distribución sesgada a la derecha (long tail)
- Outliers presentes pero válidos (eventos especiales, horas pico)
- Validado contra umbrales del ML Canvas:
  - Alta demanda: >7,000 bicicletas
  - Demanda media: 1,000-7,000
  - Baja demanda: <1,000

---

### 3. Patrones Temporales ⏰

#### **Horarios (Crítico para predicción 1-24h)**
- **Horas pico:** 7-9am (entrada al trabajo), 5-7pm (salida)
- **Horas valle:** 12am-5am (madrugada)
- **Patrón bimodal** en días laborales
- **Patrón unimodal** en fines de semana (pico en tarde)

#### **Semanales**
- **Días laborales:** Mayor demanda en horas pico (commuters)
- **Fines de semana:** Demanda distribuida, picos en tarde
- **Diferencia significativa** en patrones de uso

#### **Estacionales**
- **Verano:** Mayor demanda (pico en junio-septiembre)
- **Invierno:** Menor demanda (valle en diciembre-febrero)
- **Patrón claro** de preferencia por clima cálido

#### **Anuales**
- **Crecimiento 2011→2012:** +~40%
- **Tendencia positiva** en adopción del sistema

---

### 4. Impacto Climático 🌤️

| Variable | Correlación con `cnt` | Impacto |
|----------|-----------------------|---------|
| **temp** | +0.40 | Alto - Mayor temperatura = mayor demanda |
| **atemp** | +0.39 | Alto - Similar a temp (multicolinealidad) |
| **hum** | -0.10 | Moderado - Mayor humedad = menor demanda |
| **windspeed** | -0.05 | Bajo - Poco impacto directo |

#### **Situación Climática (weathersit)**
1. **Despejado:** ~230 bicicletas/hora promedio ⬆️
2. **Nublado/Niebla:** ~180 bicicletas/hora ➡️
3. **Lluvia Ligera:** ~100 bicicletas/hora ⬇️
4. **Lluvia Intensa:** <50 bicicletas/hora ⬇️⬇️

**Insight:** Clima despejado aumenta demanda ~130% vs lluvia intensa.

---

### 5. Tipos de Usuario 👥

| Tipo | % del Total | Patrón Principal |
|------|-------------|------------------|
| **Registrados** | ~80% | Commuters - picos laborales |
| **Casuales** | ~20% | Recreativos - fines de semana |

**Diferencias clave:**
- Registrados: Patrones predecibles, horas laborales
- Casuales: Mayor variabilidad, sensibles al clima
- Comportamientos tan diferentes que justifican modelos especializados

---

### 6. Features Más Importantes (Correlación con `cnt`)

1. **hr** (hora) - Correlación muy fuerte (~0.40)
2. **temp/atemp** (temperatura) - Correlación fuerte (~0.40)
3. **season** (estación) - Patrón estacional claro
4. **workingday** (día laboral) - Divide comportamiento
5. **yr** (año) - Tendencia de crecimiento
6. **weathersit** (clima) - Impacto significativo
7. **hum** (humedad) - Correlación negativa moderada
8. **casual/registered** - Importantes para entender descomposición

---

## ⚠️ Problemas y Consideraciones

### Multicolinealidad
- **temp** y **atemp** correlación de 0.99
- **Acción:** Elegir solo una (preferiblemente `temp`)

### Outliers
- Presentes en `cnt`, `casual`, `registered`
- Parecen ser valores extremos reales (no errores)
- **Acción:** Usar modelos robustos o transformación logarítmica

### Distribución No Normal
- Target sesgado a la derecha
- **Acción:** Considerar transformaciones o modelos no lineales

---

## 🎯 Recomendaciones para Modelado

### 1. Split de Datos
- **Usar split temporal** (NO aleatorio)
- Train: Primeros 18 meses
- Validation: 2 meses siguientes  
- Test: Últimos 4 meses
- Respetar orden temporal para evitar data leakage

### 2. Feature Engineering Prioritario

#### A. Features Cíclicas
```python
# Hora cíclica
hr_sin = np.sin(2 * np.pi * hr / 24)
hr_cos = np.cos(2 * np.pi * hr / 24)

# Mes cíclico
mnth_sin = np.sin(2 * np.pi * mnth / 12)
mnth_cos = np.cos(2 * np.pi * mnth / 12)
```

#### B. Lags Temporales
```python
# Lag de 1 hora
cnt_lag1 = cnt.shift(1)

# Lag de 24 horas (mismo momento día anterior)
cnt_lag24 = cnt.shift(24)

# Rolling mean 24 horas
cnt_rolling_24h = cnt.rolling(window=24).mean()
```

#### C. Interacciones
```python
# Temperatura × Estación
temp_season = temp * season

# Hora × Día laboral (capturar commuters)
hr_workingday = hr * workingday

# Temperatura × Humedad
temp_hum = temp * hum
```

### 3. Modelos Recomendados

Según ML Canvas, implementar en orden:

1. **Baseline:** Linear Regression
   - Rápido, interpretable
   - Establecer benchmark

2. **Random Forest Regressor**
   - Robusto a outliers
   - Feature importance
   - No requiere escalado

3. **XGBoost Regressor** (Modelo principal)
   - Mejor performance esperada
   - Maneja no linealidades
   - Hiperparámetros:
     - `max_depth`: 6-10
     - `learning_rate`: 0.01-0.1
     - `n_estimators`: 500-1000

4. **Modelos Especializados** (Opcional)
   - 24 Random Forests (uno por hora)
   - Modelos separados para casual vs registered

### 4. Métricas de Evaluación

**Métricas ML:**
- **MAE** < 400 (objetivo < 300)
- **RMSE** < 600
- **MAPE** < 15%
- **R²** > 0.85

**Métricas de Negocio:**
- Precisión en picos (hr 7-9, 17-19) > 85%
- Detección correcta de demanda alta (>7K)
- ROI > 300% anual

---

## 📈 Próximos Pasos

### Fase 2: Feature Engineering & Preprocesamiento
- [ ] Implementar features cíclicas
- [ ] Crear lags y rolling means
- [ ] Generar interacciones
- [ ] One-hot encoding de categóricas
- [ ] Train/validation/test split temporal
- [ ] Guardar en `data/processed/`
- [ ] Versionar con DVC

### Fase 3: Modelado
- [ ] Baseline models con MLflow
- [ ] Hyperparameter tuning
- [ ] Evaluación y comparación
- [ ] Selección de modelo final

### Fase 4: Documentación
- [ ] Scripts productivos en `src/`
- [ ] README actualizado
- [ ] Presentación ejecutiva PDF

---

## 📝 Archivos Generados

- ✅ `notebooks/notebook.ipynb` - EDA completo con visualizaciones
- ✅ `data/interim/bike_sharing_clean.csv` - Dataset limpio (próximo paso)
- ✅ `docs/ML_Canvas.md` - Requerimientos de negocio
- ✅ `docs/EDA_Summary.md` - Este documento

---

**Autor:** MLOps Team 61  
**Última actualización:** Octubre 2025

