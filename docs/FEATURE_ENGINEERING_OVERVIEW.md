## Feature Engineering Overview

### 1. Why We Go Beyond the Raw Columns
- `data/raw/bike_sharing_original.csv` contiene únicamente 17 variables instantáneas (clima de la hora, flags de día, etc.).  
- La demanda (`cnt`) depende fuertemente de **lo que acaba de ocurrir** (lags, tendencia diaria/semanal, volatilidad reciente).  
- Sin esa memoria temporal, modelos como Random Forest o XGBoost no tienen suficiente señal y convergen a “medias” poco útiles (ej. ~20 bikes cuando en realidad había 250).

### 2. Cómo se generan las ~50 columnas
| Paso | Archivo | Resumen |
| --- | --- | --- |
| Limpieza inicial | `src/data/data_cleaner.py` | Convierte tipos, rellena nulos, corrige rangos anómalos sin eliminar filas. |
| Feature engineering “largo” | `src/data/feature_engineering.py` | Añade lags (`cnt_transformed_lag_*`), medias móviles, volatilidades, interacciones de clima, codificaciones cíclicas, contexto histórico por hora/día, etc. |
| Almacenamiento | `data/processed/bike_sharing_features_<split>.csv` | Estos CSV contienen **todas** las features utilizadas para entrenamiento. |

> Nota: `_compute_derived_features()` en `service.py` replica solo la parte que puede calcularse con información instantánea (cíclicas, interacciones simples). Las columnas que requieren historial quedan en 0.0 si no se suministra un estado previo.

### 3. ¿Qué son los archivos `_normalized`?
- Los notebooks exploratorios (`notebooks/…`) generan versiones escaladas con `RobustScaler`.
- Se guardan en `data/processed/*_normalized.csv`; no son usadas por el pipeline de producción (entrenamiento y servicio emplean las versiones “sin normalizar”).  
- Se conservan únicamente para experimentos manuales; no confundirse con los artefactos que usa `src/models/train_model.py`.

### 4. Por qué las predicciones del servicio parecen “bajas”
- En producción, cuando llamamos a `/predict` o a `/models/{name}/{version}/predict`, solo pasamos las 12 features “básicas”.
- `_compute_derived_features()` rellena los lags/rolling/volatilidades con **0.0** porque no tiene información histórica disponible.
- Resultado: el modelo interpreta que no hubo demanda en las últimas horas → predicciones muy bajas.
- Solución: implementar un componente stateful (cache o base de datos) que calcule y suministre los lags reales antes de invocar el pipeline.

### 5. Recomendaciones prácticas
1. **Para validación realista**: toma el vector completo desde `data/processed/bike_sharing_features_<split>.csv` (50 columnas) y pásalo al modelo; así verás métricas acordes a los reportes de entrenamiento.
2. **En servicio**: hasta que no exista un mecanismo de histórico, considera limitar la API al modelo que mejor funcione sin lag (ej. una versión reducida o un modelo alternativo), o advierte a los consumidores que la respuesta será conservadora.
3. **No elimines las features extra**: son necesarias; el problema actual es la ausencia de datos históricos en producción, no la cantidad de columnas.

### 6. Ruta rápida para reproducir la ingeniería de features
```bash
python src/data/make_dataset.py        # Limpia datos y genera archivos en data/interim/
python src/features/build_features.py  # (si existiera) produce data/processed/*.csv
python src/models/train_model.py        # Entrena modelos y guarda artefactos en models/
```
*(El segundo comando puede variar según la estructura del proyecto; consulta `src/data/feature_engineering.py` y `src/models/train_model.py` para más detalle.)*

### 7. Glosario rápido
- **Lag features**: demanda de 1h, 24h, 48h… atrás.
- **Rolling stats**: promedios y desviaciones de ventanas de 3h/24h/72h.
- **Volatilidad / momentum**: cambios porcentuales y variaciones cuadráticas.
- **Historical context**: promedio histórico por hora/día, comparación contra el pasado.
- **Codificaciones cíclicas**: `hr_sin`, `hr_cos`, etc. para capturar periodicidad.


