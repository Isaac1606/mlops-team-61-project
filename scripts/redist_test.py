from src.utils.redis_manager import RedisManager
from src.models.feature_engineer_transformer import FeatureEngineerTransformer
import pandas as pd

# 1) Bootstrap con datos reales (usa tus últimas filas limpias)
df_hist = pd.read_csv("data/interim/bike_sharing_clean.csv").tail(200)
redis_mgr = RedisManager()
redis_mgr.bootstrap(df_hist)

# 2) Crear el transformer y ajustarlo
transformer = FeatureEngineerTransformer(history_window=200)
transformer.fit(df_hist, y=df_hist["cnt"])

# 3) Preparar una fila (con dteday, hr…) y transformarla
sample = df_hist.tail(1)[[
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]]
features = transformer.transform(sample)
print(features.tail(1))