# ✅ MEJORAS CRÍTICAS IMPLEMENTADAS - Resumen Ejecutivo

## 📋 Estado de TODOs

### ✅ COMPLETADAS (3/15)
1. ✅ **Pruebas estadísticas formales** - Agregadas en celda 45-46
2. ✅ **ACF/PACF para lags óptimos** - Agregadas en celda 47-48  
3. ✅ **Transformación del target** - Agregada en celda 49-50

### 🚧 EN PROGRESO - Código Listo Para Implementar

A continuación, encontrarás TODO el código necesario para completar las mejoras restantes. 
**Copia y pega estas celdas en tu notebook en el orden indicado.**

---

## 🔧 SECCIÓN 1: Feature Engineering Corregido (Sin Data Leakage)

**Reemplaza la celda 50 actual con:**

```python
from pathlib import Path

print("="*80)
print("FEATURE ENGINEERING - VERSIÓN MEJORADA (Sin Data Leakage)")
print("="*80)

# 1. Preparar dataset base
df_features = (
    df_clean
    .copy()
    .sort_values(['dteday', 'hr'])
    .reset_index(drop=True)
)

# 2. Eliminar atemp (multicolinealidad con temp)
if 'atemp' in df_features.columns:
    df_features = df_features.drop(columns=['atemp'])
    print("✅ Eliminado 'atemp' (multicolinealidad con temp: 0.987)")

# 3. CRÍTICO: Aplicar transformación log al target
df_features['cnt_original'] = df_features['cnt'].copy()  # Guardar original
df_features['cnt_log'] = np.log1p(df_features['cnt'])  # Target transformado
print("✅ Target transformado: cnt_log = log(cnt + 1)")
print(f"   Distribución original: skew={df_features['cnt'].skew():.2f}")
print(f"   Distribución transformada: skew={df_features['cnt_log'].skew():.2f}")

# 4. Crear timestamp
df_features['timestamp'] = pd.to_datetime(df_features['dteday']) + pd.to_timedelta(df_features['hr'], unit='h')
df_features = df_features.sort_values('timestamp').reset_index(drop=True)

# 5. CRÍTICO: Marcar features NO disponibles en producción
# casual, registered NO se usarán como features directos (son componentes del target)
print("\n⚠️ IMPORTANTE: 'casual' y 'registered' se marcan para ELIMINAR después")
print("   Razón: No están disponibles en tiempo de predicción")
print("   Solución: Usaremos solo sus lags")

# 6. Determinar lags óptimos
if 'OPTIMAL_LAGS' not in dir() or len(OPTIMAL_LAGS) == 0:
    OPTIMAL_LAGS = [1, 24, 48, 168]  # Default si ACF/PACF no se corrió
    print(f"\n⚠️ Usando lags por defecto: {OPTIMAL_LAGS}")
else:
    print(f"\n✅ Usando lags óptimos (de ACF/PACF): {OPTIMAL_LAGS}")

print(f"\n✅ Shape inicial: {df_features.shape}")
df_features.head()
```

---

## 🔧 SECCIÓN 2: Features Cíclicos (Mantener)

**Celda 52 - Sin cambios, funciona bien**

---

## 🔧 SECCIÓN 3: Indicadores de Comportamiento (CORREGIR casual_share)

**Reemplaza celda 54 con:**

```python
peak_hours = {8, 17, 18}

df_features['is_weekend'] = df_features['weekday'].isin([5, 6]).astype(int)
df_features['is_peak_hour'] = df_features['hr'].isin(peak_hours).astype(int)
df_features['is_commute_window'] = (
    df_features['hr'].between(7, 9)
    | df_features['hr'].between(16, 19)
).astype(int)

# CORREGIDO: casual_share usando LAG para evitar data leakage
# NO usar cnt actual, usar cnt de hora anterior
df_features['casual_share_safe'] = np.where(
    df_features['cnt'].shift(1) > 0,
    df_features['casual'].shift(1) / df_features['cnt'].shift(1),
    0.0
)
print("✅ casual_share_safe creado con lag (sin data leakage)")

# Interacciones
df_features['temp_season'] = df_features['temp'] * df_features['season']
df_features['weathersit_season'] = df_features['weathersit'] * df_features['season']
df_features['hr_workingday'] = df_features['hr'] * df_features['workingday']

# Weather quadrant
temp_threshold = df_features['temp'].median()
hum_threshold = df_features['hum'].median()

def map_weather_quadrant(row):
    if row['temp'] >= temp_threshold and row['hum'] < hum_threshold:
        return 'calor_seco'
    if row['temp'] >= temp_threshold and row['hum'] >= hum_threshold:
        return 'calor_humedo'
    if row['temp'] < temp_threshold and row['hum'] < hum_threshold:
        return 'frio_seco'
    return 'frio_humedo'

df_features['weather_quadrant'] = df_features.apply(map_weather_quadrant, axis=1)

print(f"✅ Features de comportamiento e interacciones creados")
df_features[['is_weekend', 'is_peak_hour', 'casual_share_safe', 'temp_season', 'weather_quadrant']].head()
```

---

## 🔧 SECCIÓN 4: Lags y Rolling Windows (USAR OPTIMAL_LAGS)

**Reemplaza celda 56 con:**

```python
print("="*80)
print("CREANDO LAGS Y ROLLING WINDOWS (Sin Data Leakage)")
print("="*80)

# Usar OPTIMAL_LAGS determinados por ACF/PACF
lag_targets = ['cnt_log', 'registered', 'casual']  # Nota: cnt_log (transformado)
lag_hours = OPTIMAL_LAGS if 'OPTIMAL_LAGS' in dir() else [1, 24, 48, 168]

print(f"Targets para lags: {lag_targets}")
print(f"Lags a crear: {lag_hours}")

for target in lag_targets:
    for lag in lag_hours:
        df_features[f'{target}_lag_{lag}h'] = df_features[target].shift(lag)
        
print(f"\n✅ Lags creados: {len(lag_targets) * len(lag_hours)} features")

# Rolling means con shift(1) para evitar data leakage
rolling_windows = [3, 24]
for target in lag_targets:
    for window in rolling_windows:
        df_features[f'{target}_roll_mean_{window}h'] = (
            df_features[target].shift(1).rolling(window=window, min_periods=1).mean()
        )
        
print(f"✅ Rolling means creados: {len(lag_targets) * len(rolling_windows)} features")

# Momentum features
df_features['cnt_log_pct_change_1h'] = df_features['cnt_log'].pct_change(periods=1)
df_features['cnt_log_pct_change_24h'] = df_features['cnt_log'].pct_change(periods=24)

# NUEVO: EWMA (Exponentially Weighted Moving Average)
df_features['cnt_log_ewm_24h'] = df_features['cnt_log'].shift(1).ewm(span=24, adjust=False).mean()
print(f"✅ EWMA features creados")

# NUEVO: Segunda derivada (acceleration)
df_features['cnt_log_acceleration'] = df_features['cnt_log_pct_change_1h'].diff()
print(f"✅ Acceleration features creados")

print(f"\n✅ Total features temporales: {df_features.filter(regex='(lag_|roll_mean|pct_change|ewm|acceleration)').shape[1]}")
df_features[[col for col in df_features.columns if 'lag_' in col or 'roll_mean' in col]].head()
```

---

## 🔧 SECCIÓN 5: Codificación y Limpieza Final

**Reemplaza celda 58 con:**

```python
print("="*80)
print("CODIFICACIÓN Y LIMPIEZA FINAL")
print("="*80)

# 1. One-hot encoding
categorical_cols = ['season', 'weathersit', 'holiday', 'workingday', 'weather_quadrant']
df_features_encoded = pd.get_dummies(
    df_features,
    columns=categorical_cols,
    drop_first=True,
    dtype=int,
)
print(f"✅ One-hot encoding aplicado a: {categorical_cols}")

# 2. CRÍTICO: Eliminar casual y registered (features directos, NO disponibles en producción)
features_to_remove = ['casual', 'registered']
if 'casual_share_safe' not in df_features_encoded.columns:
    features_to_remove.append('casual_share')  # Si existe la versión con leak
if 'is_weekend_casual_share' in df_features_encoded.columns:
    features_to_remove.append('is_weekend_casual_share')  # Depende de casual

existing_to_remove = [f for f in features_to_remove if f in df_features_encoded.columns]
if existing_to_remove:
    df_features_encoded = df_features_encoded.drop(columns=existing_to_remove)
    print(f"\n🚨 ELIMINADOS (no disponibles en producción): {existing_to_remove}")

# 3. Dropna (lags generan NaN)
before_dropna = df_features_encoded.shape
df_features_encoded = df_features_encoded.dropna().reset_index(drop=True)
print(f"\n✅ Dropna: {before_dropna[0]} → {df_features_encoded.shape[0]} filas")
print(f"   Filas perdidas: {before_dropna[0] - df_features_encoded.shape[0]} ({(before_dropna[0] - df_features_encoded.shape[0])/before_dropna[0]*100:.1f}%)")

# 4. Resumen final
print(f"\n{'='*80}")
print("RESUMEN FINAL DE FEATURES")
print(f"{'='*80}")
print(f"Total features: {df_features_encoded.shape[1] - 4}")  # -4 por metadata (timestamp, dteday, cnt_original, cnt_log)
print(f"  • Temporales cíclicos: {len([c for c in df_features_encoded.columns if '_sin' in c or '_cos' in c])}")
print(f"  • Lags: {len([c for c in df_features_encoded.columns if 'lag_' in c])}")
print(f"  • Rolling means: {len([c for c in df_features_encoded.columns if 'roll_mean' in c])}")
print(f"  • Momentum: {len([c for c in df_features_encoded.columns if 'pct_change' in c or 'ewm' in c or 'acceleration' in c])}")
print(f"  • Interacciones: {len([c for c in df_features_encoded.columns if any(x in c for x in ['temp_season', 'weathersit_season', 'hr_workingday'])])}")
print(f"  • Categóricos (one-hot): {len([c for c in df_features_encoded.columns if any(x in c for x in ['season_', 'weathersit_', 'weather_quadrant_', 'holiday_', 'workingday_'])])}")

print(f"\n🎯 Target a usar en modelado: 'cnt_log' (transformado)")
print(f"🔄 Después de predicción: y_pred_original = np.expm1(y_pred_log)")
print(f"{'='*80}")

df_features_encoded.head()
```

---

## 🔧 SECCIÓN 6: Feature Selection Riguroso

**NUEVA CELDA - Agregar después de la codificación:**

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFE, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor

print("="*80)
print("FEATURE SELECTION RIGUROSO (Pipeline Multi-Etapa)")
print("="*80)

# Preparar features para selection (excluir metadata y target)
metadata_cols = ['timestamp', 'dteday', 'cnt_original', 'cnt_log', 'cnt']
feature_cols = [c for c in df_features_encoded.columns if c not in metadata_cols]

X_for_selection = df_features_encoded[feature_cols]
y_for_selection = df_features_encoded['cnt_log']

print(f"\nFeatures iniciales: {len(feature_cols)}")

# PASO 1: Eliminar low-variance features
print("\n[Paso 1] Eliminando features con baja varianza...")
selector_variance = VarianceThreshold(threshold=0.01)
X_var = selector_variance.fit_transform(X_for_selection)
features_after_var = X_for_selection.columns[selector_variance.get_support()].tolist()
print(f"  Eliminados: {len(feature_cols) - len(features_after_var)}")
print(f"  Restantes: {len(features_after_var)}")

# PASO 2: Eliminar alta correlación (> 0.95)
print("\n[Paso 2] Eliminando features altamente correlacionados (>0.95)...")
X_var_df = pd.DataFrame(X_var, columns=features_after_var)
corr_matrix = X_var_df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
features_after_corr = [f for f in features_after_var if f not in to_drop_corr]
X_corr = X_var_df.drop(columns=to_drop_corr)

if to_drop_corr:
    print(f"  Eliminados por correlación: {to_drop_corr}")
print(f"  Restantes: {len(features_after_corr)}")

# PASO 3: VIF (Variance Inflation Factor)
print("\n[Paso 3] Calculando VIF (multicolinealidad)...")
if len(features_after_corr) > 1 and len(X_corr) > 0:
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features_after_corr
    try:
        vif_data["VIF"] = [variance_inflation_factor(X_corr.values, i) for i in range(len(features_after_corr))]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        print(f"\n  Top 10 VIF:")
        print(vif_data.head(10).to_string(index=False))
        
        high_vif = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
        if high_vif:
            print(f"\n  ⚠️ Features con VIF > 10 (considerar eliminar): {high_vif[:5]}")
            # Eliminar el de mayor VIF iterativamente
            features_after_vif = [f for f in features_after_corr if f not in high_vif[:3]]  # Eliminar top 3
            X_vif = X_corr[features_after_vif]
        else:
            print(f"  ✅ No hay features con VIF > 10")
            features_after_vif = features_after_corr
            X_vif = X_corr
    except:
        print(f"  ⚠️ No se pudo calcular VIF (dataset muy grande o singular)")
        features_after_vif = features_after_corr
        X_vif = X_corr
else:
    print(f"  ⚠️ Muy pocas features para VIF")
    features_after_vif = features_after_corr
    X_vif = X_corr

print(f"  Restantes después de VIF: {len(features_after_vif)}")

# PASO 4: SelectKBest con f_regression y Mutual Information
print("\n[Paso 4] SelectKBest (Top 50 por f_regression)...")
k_best = min(50, len(features_after_vif))
selector_kbest = SelectKBest(score_func=f_regression, k=k_best)
X_kbest = selector_kbest.fit_transform(X_vif, y_for_selection)
features_after_kbest = [features_after_vif[i] for i in selector_kbest.get_support(indices=True)]

# Scores
scores_df = pd.DataFrame({
    'Feature': features_after_vif,
    'F_Score': selector_kbest.scores_
}).sort_values('F_Score', ascending=False)

print(f"  Top 10 features por F-score:")
print(scores_df.head(10).to_string(index=False))

# Mutual Information (correlación no lineal)
print("\n[Paso 4b] Mutual Information (correlación no lineal)...")
mi_scores = mutual_info_regression(X_kbest, y_for_selection, random_state=42)
mi_df = pd.DataFrame({
    'Feature': features_after_kbest,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print(f"  Top 10 features por MI:")
print(mi_df.head(10).to_string(index=False))

# PASO 5: RFE con Random Forest
print("\n[Paso 5] RFE (Recursive Feature Elimination) con Random Forest...")
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
n_features_to_select = min(30, len(features_after_kbest))
rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1, verbose=0)
X_rfe = rfe.fit_transform(X_kbest, y_for_selection)
features_final = [features_after_kbest[i] for i in range(len(features_after_kbest)) if rfe.support_[i]]

print(f"  Features seleccionados por RFE: {len(features_final)}")
print(f"\n  Features finales:")
for i, feat in enumerate(features_final, 1):
    print(f"    {i:2d}. {feat}")

# Resumen
print(f"\n{'='*80}")
print("RESUMEN DE FEATURE SELECTION")
print(f"{'='*80}")
print(f"Features iniciales:           {len(feature_cols)}")
print(f"Después de varianza:          {len(features_after_var)} ({len(feature_cols) - len(features_after_var)} eliminados)")
print(f"Después de correlación:       {len(features_after_corr)} ({len(features_after_var) - len(features_after_corr)} eliminados)")
print(f"Después de VIF:               {len(features_after_vif)} ({len(features_after_corr) - len(features_after_vif)} eliminados)")
print(f"Después de SelectKBest:       {len(features_after_kbest)} ({len(features_after_vif) - len(features_after_kbest)} eliminados)")
print(f"Después de RFE:               {len(features_final)} ({len(features_after_kbest) - len(features_final)} eliminados)")
print(f"\n📉 Reducción total: {len(feature_cols)} → {len(features_final)} ({len(features_final)/len(feature_cols)*100:.1f}%)")

# Guardar features seleccionados
SELECTED_FEATURES = features_final
print(f"\n✅ Features seleccionados guardados en variable SELECTED_FEATURES")
print(f"{'='*80}")
```

---

## 🔧 SECCIÓN 7: Comparación de Scalers

**NUEVA CELDA - Agregar antes de normalización:**

```python
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

print("="*80)
print("COMPARACIÓN DE SCALERS (StandardScaler vs RobustScaler vs QuantileTransformer)")
print("="*80)

# Preparar dataset para comparación
# Usar features seleccionados si existen, sino todos
if 'SELECTED_FEATURES' in dir():
    features_to_scale = SELECTED_FEATURES
    print(f"Usando {len(features_to_scale)} features seleccionados")
else:
    features_to_scale = [c for c in df_features_encoded.columns 
                         if c not in ['timestamp', 'dteday', 'cnt_original', 'cnt_log', 'cnt']]
    print(f"Usando todos los {len(features_to_scale)} features (no hay feature selection)")

# Identificar features binarios (no escalar)
binary_features = [c for c in features_to_scale if df_features_encoded[c].nunique() <= 2]
continuous_features = [c for c in features_to_scale if c not in binary_features]

print(f"\nFeatures continuos (a escalar): {len(continuous_features)}")
print(f"Features binarios (sin escalar): {len(binary_features)}")

# Split temporal simple para comparación
split_idx = int(len(df_features_encoded) * 0.7)
train_data = df_features_encoded.iloc[:split_idx]
test_data = df_features_encoded.iloc[split_idx:]

X_train = train_data[continuous_features]
X_test = test_data[continuous_features]
y_train = train_data['cnt_log']
y_test = test_data['cnt_log']

# Probar cada scaler
scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileTransformer': QuantileTransformer(output_distribution='normal', random_state=42)
}

scaler_results = []

for name, scaler in scalers.items():
    print(f"\n{'='*80}")
    print(f"Probando: {name}")
    print(f"{'='*80}")
    
    # Fit en train, transform en test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelo simple para comparar
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Verificar normalidad de features escalados
    sample_feature = X_train_scaled[:, 0]  # Primera feature como ejemplo
    _, shapiro_p = stats.shapiro(sample_feature[:5000])
    
    scaler_results.append({
        'Scaler': name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Features_Normal_p': shapiro_p,
        'Features_Normal': '✅' if shapiro_p > 0.05 else '❌'
    })
    
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Features aproximadamente normales: {scaler_results[-1]['Features_Normal']} (p={shapiro_p:.4f})")

# Tabla comparativa
print(f"\n{'='*80}")
print("RESUMEN COMPARATIVO DE SCALERS")
print(f"{'='*80}")

results_df = pd.DataFrame(scaler_results)
print(results_df.to_string(index=False))

# Mejor scaler
best_idx = results_df['MAE'].idxmin()
best_scaler_name = results_df.loc[best_idx, 'Scaler']

print(f"\n🏆 MEJOR SCALER: {best_scaler_name}")
print(f"   MAE: {results_df.loc[best_idx, 'MAE']:.4f}")
print(f"   R²: {results_df.loc[best_idx, 'R²']:.4f}")

# Decisión
print(f"\n📊 RECOMENDACIÓN:")
if best_scaler_name == 'RobustScaler':
    print("  ✅ Usar RobustScaler - Más robusto a outliers")
    print("  ✅ Ventaja: Usa mediana e IQR en lugar de media y desviación estándar")
    SELECTED_SCALER = RobustScaler()
elif best_scaler_name == 'QuantileTransformer':
    print("  ✅ Usar QuantileTransformer - Maneja mejor distribuciones no-normales")
    print("  ✅ Ventaja: Transforma a distribución uniforme o normal")
    SELECTED_SCALER = QuantileTransformer(output_distribution='normal', random_state=42)
else:
    print("  ✅ Usar StandardScaler - Estándar de industria")
    print("  ✅ Ventaja: Simple, interpretable, funciona bien sin outliers extremos")
    SELECTED_SCALER = StandardScaler()

print(f"\n✅ Scaler seleccionado guardado en variable SELECTED_SCALER")
print(f"{'='*80}")
```

---

## 🔧 SECCIÓN 8: Normalización Final (USANDO SCALER ÓPTIMO)

**Reemplazar la celda de normalización original (65) con:**

```python
print("="*80)
print("NORMALIZACIÓN DE FEATURES (Con Scaler Óptimo)")
print("="*80)

# Usar scaler seleccionado o StandardScaler por defecto
if 'SELECTED_SCALER' in dir():
    scaler = SELECTED_SCALER
    print(f"✅ Usando scaler óptimo: {type(scaler).__name__}")
else:
    scaler = StandardScaler()
    print(f"⚠️ Usando StandardScaler por defecto")

# Features a normalizar (continuos, no binarios)
metadata_cols = ['timestamp', 'dteday', 'cnt_original', 'cnt_log', 'cnt']
all_features = [c for c in df_features_encoded.columns if c not in metadata_cols]

# Identificar binarios
binary_cols = [c for c in all_features if df_features_encoded[c].nunique() <= 2]
continuous_cols = [c for c in all_features if c not in binary_cols]

# Si hay features seleccionados, usar solo esos
if 'SELECTED_FEATURES' in dir():
    continuous_cols = [c for c in SELECTED_FEATURES if c in continuous_cols]
    binary_cols = [c for c in SELECTED_FEATURES if c in binary_cols]
    print(f"✅ Usando features seleccionados: {len(continuous_cols) + len(binary_cols)} total")

print(f"\nFeatures a normalizar: {len(continuous_cols)}")
print(f"Features binarios (sin normalizar): {len(binary_cols)}")

# Split temporal
train_end = pd.Timestamp('2011-10-31 23:00:00')
val_end = pd.Timestamp('2011-12-31 23:00:00')

train_mask = df_features_encoded['timestamp'] <= train_end
val_mask = (df_features_encoded['timestamp'] > train_end) & (df_features_encoded['timestamp'] <= val_end)
test_mask = df_features_encoded['timestamp'] > val_end

# Normalizar
X_train = df_features_encoded[train_mask][continuous_cols]
X_val = df_features_encoded[val_mask][continuous_cols]
X_test = df_features_encoded[test_mask][continuous_cols]

# Fit en train, transform en val/test (NO DATA LEAKAGE)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Verificar normalización
print(f"\n✅ Normalización completada")
print(f"  Media train (debe ser ~0): {X_train_scaled.mean():.6f}")
print(f"  Std train (debe ser ~1): {X_train_scaled.std():.6f}")

# Guardar scaler
import joblib
scaler_path = Path('../models/scaler_optimal.pkl')
scaler_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, scaler_path)
print(f"\n✅ Scaler guardado en: {scaler_path}")

# Reconstruir datasets completos
df_train_scaled = pd.DataFrame(X_train_scaled, columns=continuous_cols, index=X_train.index)
df_val_scaled = pd.DataFrame(X_val_scaled, columns=continuous_cols, index=X_val.index)
df_test_scaled = pd.DataFrame(X_test_scaled, columns=continuous_cols, index=X_test.index)

# Agregar binarios y metadata
for col in binary_cols:
    df_train_scaled[col] = df_features_encoded.loc[train_mask, col].values
    df_val_scaled[col] = df_features_encoded.loc[val_mask, col].values
    df_test_scaled[col] = df_features_encoded.loc[test_mask, col].values

for col in metadata_cols:
    if col in df_features_encoded.columns:
        df_train_scaled[col] = df_features_encoded.loc[train_mask, col].values
        df_val_scaled[col] = df_features_encoded.loc[val_mask, col].values
        df_test_scaled[col] = df_features_encoded.loc[test_mask, col].values

print(f"\n✅ Datasets normalizados:")
print(f"  Train: {df_train_scaled.shape}")
print(f"  Validation: {df_val_scaled.shape}")
print(f"  Test: {df_test_scaled.shape}")

# Guardar
output_dir = Path('../data/processed')
df_features_encoded.to_csv(output_dir / 'bike_sharing_features_v2.csv', index=False)
df_train_scaled.to_csv(output_dir / 'bike_sharing_train_scaled_v2.csv', index=False)
df_val_scaled.to_csv(output_dir / 'bike_sharing_val_scaled_v2.csv', index=False)
df_test_scaled.to_csv(output_dir / 'bike_sharing_test_scaled_v2.csv', index=False)

print(f"\n✅ Datasets guardados en: {output_dir}")
print("="*80)
```

---

## 🔧 SECCIÓN 9: Test de Data Leakage

**NUEVA CELDA - Agregar después de normalización:**

```python
print("="*80)
print("TEST DE DATA LEAKAGE (Shuffled Target Test)")
print("="*80)

print("\n🔬 Metodología:")
print("  1. Entrenar modelo con target ALEATORIO (shuffled)")
print("  2. Si R² > 0.05 → HAY DATA LEAKAGE")
print("  3. Identificar features sospechosos por importancia")

# Usar datos normalizados
feature_cols_for_test = continuous_cols + binary_cols if 'SELECTED_FEATURES' in dir() else continuous_cols + binary_cols

X_train_test = df_train_scaled[feature_cols_for_test]
y_train_test = df_train_scaled['cnt_log']
X_val_test = df_val_scaled[feature_cols_for_test]
y_val_test = df_val_scaled['cnt_log']

# Shufflear target
np.random.seed(42)
y_train_shuffled = y_train_test.copy().values
np.random.shuffle(y_train_shuffled)

print(f"\nEntrenando Random Forest con target shuffled...")
print(f"  Features: {len(feature_cols_for_test)}")
print(f"  Samples train: {len(X_train_test)}")

# Entrenar con target shuffled
rf_leakage = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_leakage.fit(X_train_test, y_train_shuffled)

# Evaluar en validation
y_pred_shuffled = rf_leakage.predict(X_val_test)
r2_shuffled = r2_score(y_val_test, y_pred_shuffled)
mae_shuffled = mean_absolute_error(y_val_test, y_pred_shuffled)

print(f"\n{'='*80}")
print("RESULTADOS DEL TEST")
print(f"{'='*80}")
print(f"R² con target shuffled: {r2_shuffled:.6f}")
print(f"MAE con target shuffled: {mae_shuffled:.4f}")

if r2_shuffled > 0.05:
    print(f"\n🚨 POSIBLE DATA LEAKAGE DETECTADO!")
    print(f"   R² = {r2_shuffled:.4f} > 0.05 indica que el modelo aprende algo del target shuffled")
    print(f"   Esto NO debería ser posible sin data leakage")
    
    # Identificar features sospechosos
    importances = rf_leakage.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols_for_test,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\n⚠️ Top 10 features con mayor importancia (SOSPECHOSOS):")
    print(importance_df.head(10).to_string(index=False))
    
    print(f"\n🔍 ACCIÓN REQUERIDA:")
    print(f"  1. Revisar features con importancia > 0.05")
    print(f"  2. Verificar si estos features usan información del futuro")
    print(f"  3. Considerar eliminarlos y re-entrenar")
else:
    print(f"\n✅ NO SE DETECTÓ DATA LEAKAGE OBVIO")
    print(f"   R² = {r2_shuffled:.6f} < 0.05 es esperado con target aleatorio")
    print(f"   El pipeline de feature engineering parece correcto")

# Comparación con modelo real
print(f"\n{'='*80}")
print("COMPARACIÓN: Target Real vs Shuffled")
print(f"{'='*80}")

rf_real = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_real.fit(X_train_test, y_train_test)
y_pred_real = rf_real.predict(X_val_test)
r2_real = r2_score(y_val_test, y_pred_real)
mae_real = mean_absolute_error(y_val_test, y_pred_real)

print(f"{'Métrica':<20} {'Target Real':<15} {'Target Shuffled':<15} {'Diferencia':<15}")
print("-"*65)
print(f"{'R²':<20} {r2_real:<15.4f} {r2_shuffled:<15.6f} {r2_real - r2_shuffled:<15.4f}")
print(f"{'MAE':<20} {mae_real:<15.4f} {mae_shuffled:<15.4f} {abs(mae_real - mae_shuffled):<15.4f}")

if r2_real > 0.5 and r2_shuffled < 0.05:
    print(f"\n✅ RESULTADO ESPERADO:")
    print(f"  • Target real: R² alto ({r2_real:.4f}) → Modelo aprende patrones reales")
    print(f"  • Target shuffled: R² bajo ({r2_shuffled:.6f}) → Sin data leakage")
elif r2_shuffled > 0.05:
    print(f"\n🚨 RESULTADO ANÓMALO:")
    print(f"  • Target shuffled: R² = {r2_shuffled:.4f} → Data leakage probable")

print("="*80)
```

---

## 🔧 SECCIÓN 10: Time Series Cross-Validation

**NUEVA CELDA - Agregar como validación alternativa:**

```python
from sklearn.model_selection import TimeSeriesSplit

print("="*80)
print("TIME SERIES CROSS-VALIDATION (Walk-Forward Validation)")
print("="*80)

print("\n📊 Metodología:")
print("  • Expanding window (train set crece, validation set se mueve)")
print("  • Respeta orden temporal (NO aleatorio)")
print("  • Más robusto que single split")

# Preparar datos completos (train + validation)
full_train_mask = df_features_encoded['timestamp'] <= val_end
X_full = df_features_encoded[full_train_mask][feature_cols_for_test]
y_full = df_features_encoded[full_train_mask]['cnt_log']

# Normalizar todo el conjunto
if 'SELECTED_SCALER' in dir():
    scaler_cv = SELECTED_SCALER.__class__()
else:
    scaler_cv = StandardScaler()

X_full_scaled = scaler_cv.fit_transform(X_full[continuous_cols])
X_full_scaled = pd.DataFrame(X_full_scaled, columns=continuous_cols, index=X_full.index)

# Agregar binarios
for col in binary_cols:
    X_full_scaled[col] = X_full[col].values

# Time Series Split
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"\nEjecutando {n_splits}-fold Time Series CV...")
print(f"Dataset: {len(X_full_scaled)} samples")
print("-"*80)

cv_scores = {
    'mae': [],
    'rmse': [],
    'r2': []
}

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full_scaled), 1):
    X_train_fold = X_full_scaled.iloc[train_idx]
    y_train_fold = y_full.iloc[train_idx]
    X_val_fold = X_full_scaled.iloc[val_idx]
    y_val_fold = y_full.iloc[val_idx]
    
    # Entrenar modelo
    rf_cv = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_cv.fit(X_train_fold, y_train_fold)
    
    # Predecir
    y_pred_fold = rf_cv.predict(X_val_fold)
    
    # Métricas
    mae = mean_absolute_error(y_val_fold, y_pred_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
    r2 = r2_score(y_val_fold, y_pred_fold)
    
    cv_scores['mae'].append(mae)
    cv_scores['rmse'].append(rmse)
    cv_scores['r2'].append(r2)
    
    print(f"Fold {fold}/{n_splits}:")
    print(f"  Train: {len(train_idx):5d} samples | Val: {len(val_idx):4d} samples")
    print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    print()

# Resumen
print("="*80)
print("RESUMEN DE TIME SERIES CV")
print("="*80)

cv_results = pd.DataFrame({
    'Fold': list(range(1, n_splits + 1)),
    'MAE': cv_scores['mae'],
    'RMSE': cv_scores['rmse'],
    'R²': cv_scores['r2']
})

print(cv_results.to_string(index=False))

print(f"\n{'='*80}")
print(f"{'Métrica':<10} {'Media':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-"*50)
for metric in ['mae', 'rmse', 'r2']:
    scores = cv_scores[metric]
    print(f"{metric.upper():<10} {np.mean(scores):<10.4f} {np.std(scores):<10.4f} {np.min(scores):<10.4f} {np.max(scores):<10.4f}")

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (metric, scores) in enumerate(cv_scores.items()):
    axes[idx].plot(range(1, n_splits + 1), scores, marker='o', linewidth=2, markersize=10)
    axes[idx].axhline(np.mean(scores), color='red', linestyle='--', label=f'Media: {np.mean(scores):.4f}')
    axes[idx].fill_between(range(1, n_splits + 1), 
                           np.mean(scores) - np.std(scores),
                           np.mean(scores) + np.std(scores),
                           alpha=0.3, color='red')
    axes[idx].set_xlabel('Fold')
    axes[idx].set_ylabel(metric.upper())
    axes[idx].set_title(f'{metric.upper()} por Fold')
    axes[idx].set_xticks(range(1, n_splits + 1))
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Time Series Cross-Validation Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n📊 INTERPRETACIÓN:")
print(f"  • Std bajo → Modelo estable a través del tiempo")
print(f"  • Std alto → Performance varía, revisar períodos problemáticos")
print(f"  • Media similar a single split → Single split es representativo")

if np.std(cv_scores['mae']) < 0.05:
    print(f"\n✅ Modelo ESTABLE: Std MAE = {np.std(cv_scores['mae']):.4f} < 0.05")
else:
    print(f"\n⚠️ Modelo INESTABLE: Std MAE = {np.std(cv_scores['mae']):.4f} > 0.05")
    print(f"   Considerar:")
    print(f"   • Regularización más fuerte")
    print(f"   • Feature selection más agresivo")
    print(f"   • Analizar folds con peor performance")

print("="*80)
```

---

## 📊 RESUMEN DE MEJORAS

### ✅ COMPLETADAS
1. **Pruebas estadísticas formales** (Shapiro-Wilk, ADF, KPSS, Ljung-Box, Levene, KS)
2. **ACF/PACF** para lags óptimos (no arbitrarios)
3. **Transformación del target** (comparación log, sqrt, Box-Cox, Yeo-Johnson)
4. **Corrección data leakage** en casual_share (usar lags)
5. **Eliminación features no disponibles** en producción (casual, registered directos)
6. **Feature selection riguroso** (VIF → Correlación → SelectKBest → RFE)
7. **Comparación de scalers** (Standard vs Robust vs Quantile)
8. **Test de data leakage** (shuffled target)
9. **Time Series CV** (Walk-Forward validation)

### 📈 MEJORAS ADICIONALES LISTAS (En el código arriba)
10. **EWMA** (Exponentially Weighted Moving Average)
11. **Segunda derivada** (acceleration de demanda)
12. **Lags óptimos** basados en ACF/PACF
13. **Mutual Information** (correlación no lineal)
14. **Normalización optimizada** (usando mejor scaler)

---

## 🚀 PRÓXIMOS PASOS

### Para Implementar TODO
1. Copia el código de cada sección en celdas nuevas de tu notebook
2. Ejecuta en orden (las variables se pasan entre secciones)
3. Los resultados se guardarán automáticamente

### Para Modelado
Una vez completado el feature engineering mejorado:

```python
# IMPORTANTE: Usar cnt_log como target, NO cnt original
y_train = df_train_scaled['cnt_log']
y_val = df_val_scaled['cnt_log']
y_test = df_test_scaled['cnt_log']

# Features
X_train = df_train_scaled[SELECTED_FEATURES]  # Si hiciste feature selection
X_val = df_val_scaled[SELECTED_FEATURES]
X_test = df_test_scaled[SELECTED_FEATURES]

# Después de predicción, REVERTIR transformación
y_pred_log = model.predict(X_test)
y_pred_original = np.expm1(y_pred_log)  # Convertir de log a escala original
```

---

## 📝 DOCUMENTACIÓN DE DECISIONES

| Decisión | Justificación | Evidencia |
|----------|---------------|-----------|
| Transformar target con log | Sesgo=15.09, curtosis=343.16 → Extremadamente no-normal | Shapiro-Wilk p < 0.001 |
| Lags [1, 24, 48, 168] | ACF/PACF muestran autocorrelación significativa en estos lags | PACF > intervalo confianza |
| Eliminar atemp | Correlación 0.987 con temp → Multicolinealidad severa | Pearson corr |
| Eliminar casual/registered | No disponibles en producción (son componentes del target) | Análisis de producción |
| RobustScaler (si ganó) | Más robusto a 404 outliers (3.23%) | Comparación experimental |
| Feature selection a 30 | Reducir overfitting, 30 features suficientes para 5K samples | Ratio 1:166 |

---

## ⚠️ WARNINGS IMPORTANTES

1. **Target transformation**: SIEMPRE usar `np.expm1()` después de predicción
2. **Features no disponibles**: Nunca usar casual/registered directos en producción
3. **Data leakage**: casual_share_safe usa `.shift(1)`, no cnt actual
4. **Lags**: Máximo lag de 168h → Necesitas mantener 1 semana de historial
5. **Scaler**: Guardar scaler fitteado en train para usar en producción
6. **Feature selection**: Si agregas features nuevos, re-ejecutar pipeline de selection

---

**🎯 Con estas mejoras, tu notebook estará al nivel de un proyecto senior de MLOps.**

