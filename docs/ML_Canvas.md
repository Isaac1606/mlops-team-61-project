# ML Canvas - Bike Sharing Demand Prediction

## 📊 PREDICTION TASK

**Tipo:** Regresión multivariada para predicción numérica de demanda.

**Entidad:** Intervalos de tiempo específicos (hora del día).

**Outcome:** Variable objetivo `cnt` (cantidad total bicicletas rentadas), rango 1-8,000+ por hora.

**Temporalidad:** Predicciones para próximas 1-24 horas. Observación en tiempo real mediante sensores automáticos del sistema. Datos disponibles minutos después de cada período.

**Granularidad:** Horaria (0-23 horas) o diaria. Horizonte: corto plazo (1-24h) para operación; mediano plazo (7-30 días) para planificación.

---

## 🎯 DECISIONS

**Redistribución:** Si predicción > 7,000: aumentar disponibilidad en estaciones. Si < 1,000: concentrar flota.

**Mantenimiento:** Demanda baja predecida → programar mantenimiento 24-48h adelante.

**Personal:** Predicción alta → más staff. Baja → personal reducido.

**Pricing dinámico:** Demanda alta → tarifas aumentadas. Baja → promociones/descuentos.

**Alertas:** Activar si predicción > 80% capacidad para rebalanceo automático.

**Parámetros ajustables:** Umbrales de demanda, ventana de decisión (0-2 horas), factores externos (eventos, clima).

---

## 💎 VALUE PROPOSITION

**Beneficiarios:** Operadores (reducir costos), usuarios (disponibilidad predecible), municipios (planeación urbana).

### Problemas abordados:
- **Ineficiencia operativa:** rebalanceo manual → predicción automática (-20% costos)
- **Insatisfacción:** estaciones vacías → disponibilidad garantizada (-30% viajes fallidos)
- **Ingresos subóptimos:** tarifa fija → pricing dinámico (+10-15% revenue)

**Integración:** Dashboard operativo, API REST, app móvil, webhooks para automatización.

---

## 📥 DATA COLLECTION

**Fuentes iniciales:** Sensores bikeshare (conteos horarios), registros transaccionales, APIs clima, calendario feriados, eventos urbanos.

**Actualización:** Datos operativos cada hora, clima cada 3 horas, eventos semanales.

**Control de costos:** Almacenamiento comprimido (500MB), suscripciones API anuales (~$2K), procesamiento batch nocturno.

**Freshness:** Validación automática cada 6 horas. Reentrenamiento mensual con datos últimos 30 días.

**Labeling:** Variables objetivo etiquetadas automáticamente. Validación manual mensual en 1% de datos.

### 🔍 Umbrales de Validación de Datos

Para garantizar calidad y consistencia, se establecieron los siguientes umbrales de validación:

**Variables de Demanda:**
- `cnt` (total): 0 - 10,000 bicicletas/hora
- `casual` (usuarios casuales): 0 - 3,000 usuarios/hora  
- `registered` (usuarios registrados): 0 - 8,000 usuarios/hora

**Justificación:** Basado en análisis IQR y alineación con umbrales de decisión del ML Canvas (demanda alta > 7,000). Valores fuera de estos rangos se consideran errores de sistema o corrupciones de datos y se eliminan durante el preprocesamiento.

**Impacto:** Durante la limpieza inicial, se eliminaron ~5,155 filas (29% del dataset) que contenían valores nulos o fuera de rangos válidos, reteniendo 12,571 observaciones horarias de alta calidad.

---

## 🗄️ DATA SOURCES

### Internas:
- **Tabla bike_trips:** trip_id, estaciones, timestamps, user_type
- **Tabla station_inventory:** fecha_hora, bicicletas disponibles, espacios
- **DW daily_demand:** FACT con casual/registered/cnt; DIM temporal

### Externas:
- **OpenWeatherMap API:** temp, humedad, viento (horario)
- **Google Calendar:** feriados locales/federales
- **Eventbrite/APIs eventos:** nombre, ubicación, fecha, asistencia esperada

---

## 💰 IMPACT SIMULATION

**Matriz costos:** 
- Demanda alta correcta: +$4,500
- Demanda baja correcta: +$800
- Predicción alta fallida: -$1,200
- Predicción baja fallida: -$600

**Dataset validación:** Últimas 8 semanas (20% datos), no usado en entrenamiento.

**Métricas objetivo:**
- MAE < 400 bicicletas/hora (target: <300)
- MAPE < 15%
- ROI > 300% anual

**Deployment criteria:**
- MAE < 400
- Precisión picos > 85%
- ROI > 250%

**Fairness:** Equidad geográfica (todas zonas), igualdad de usuarios, auditoría mensual de sesgo.

---

## 🔮 MAKING PREDICTIONS

**Modo:** Hybrid
- Batch diaria (2:00 AM, 7 días adelante)
- Real-time cada 15min (próximas 3 horas)

**Latencia:**
- Batch ≤ 30min
- Real-time ≤ 5seg
- API ad-hoc ≤ 2seg

**SLAs:** 99% batch, 99.9% real-time

**Recursos:** Servidor batch (8 cores, 32GB), Kubernetes real-time (2-5 replicas auto-scaling), PostgreSQL+Redis.

**Stack:** Python, scikit-learn, XGBoost, Apache Airflow, Docker.

---

## 🤖 BUILDING MODELS

### Modelos en producción (3 especializados):
1. **Global:** XGBoost Regressor
2. **24 horarios:** Random Forest (uno por hora)
3. **Anomalía:** Isolation Forest

**Update strategy:**
- Mensual: modelos primario/secundario
- Semanal: detección de anomalías

**Triggering retraining:**
- 1er mes: automático
- Degradación RMSE > 15%
- Eventos especiales

**Recursos:** GPU T4, 60min máximo, 4vCPU, 16GB RAM

**Personal:** 0.5 FTE Data Scientist, 0.25 FTE ML Engineer

---

## 🔧 FEATURES

### Input: 13 features base

**Categóricas:** season, mnth, hr, weekday, weathersit (one-hot encoding)

**Binarias:** holiday, workingday, yr

**Continuas:** temp, atemp, hum, windspeed (normalizadas 0-1)

### Transformaciones:
- **Ciclicidad:** sin/cos para horas/meses
- **Agregaciones:** media móvil 7 días, lags (t-1, t-24)
- **Interacciones:** temp×hum, season×temp, hr×weekend
- **Normalización:** StandardScaler, feature selection automática

---

## 📈 MONITORING

### Métricas ML:
- MAE (umbral < 400)
- RMSE (< 600)
- MAPE (< 15%)
- R² (> 0.85)

### KPIs de negocio:
- Ahorro costos: 20%
- Satisfacción usuario: 98% viajes exitosos
- Revenue: +12%
- Uptime: 99.5%

### Frecuencia:
- **Diaria:** MAE/RMSE
- **Semanal:** KPIs
- **Mensual:** Reporte ejecutivo
- **Trimestral:** Auditoría completa

### Alertas:
- **Crítica:** MAE > 600
- **Warning:** MAE > 500
- **Info:** Retrain completado

**Dashboard:** Real-time accuracy, operacional, negocio, sistema, anomalías.

---

## 📝 Metadata

- **Versión:** 1.0
- **Fecha:** Octubre 2025
- **Proyecto:** MLOps Team 61 - Bike Sharing Prediction
- **Dataset:** Capital Bikeshare (2011-2012)

