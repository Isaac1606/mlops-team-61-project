"""Utility class for Redis operations over cnt history."""

from __future__ import annotations

import os
from typing import List
from datetime import datetime

import pandas as pd
import numpy as np
from redis import Redis


class RedisManager:
    """Wrapper around redis-py for storing cnt_transformed history."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
    ) -> None:
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db or int(os.getenv("REDIS_DB", "0"))
        self.redis = Redis(host=self.host, port=self.port, db=self.db, decode_responses=False)
        self.observed_key = os.getenv("REDIS_OBSERVED_KEY", "TS:GLOBAL:HOURLY_CNT_TF:OBSERVED")
        self.predicted_key = os.getenv("REDIS_PREDICTED_KEY", "TS:GLOBAL:HOURLY_CNT_TF:PREDICTED")
        self.cache_key = os.getenv("REDIS_PREDICTION_CACHE_KEY", "PREDICTIONS:CACHE")

    # ------------------------------------------------------------------
    # Observed series (ground truth)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_time(value: float | int | datetime) -> float | str:
        if isinstance(value, datetime):
            return value.timestamp()
        value_float = float(value)
        if np.isposinf(value_float):
            return "+inf"
        if np.isneginf(value_float):
            return "-inf"
        return value_float

    def write_observed(self, value: float, timestamp: float | int | datetime) -> None:
        ts = self._normalize_time(timestamp)
        self.redis.zadd(self.observed_key, {str(value).encode("utf-8"): ts})

    def observed_exists(self, timestamp: float | int | datetime) -> bool:
        ts = self._normalize_time(timestamp)
        result = self.redis.zrangebyscore(self.observed_key, ts, ts, start=0, num=1)
        return bool(result)

    def read_observed(self, start_ts: float | int | datetime, end_ts: float | int | datetime) -> List[tuple[bytes, float]]:
        start = self._normalize_time(start_ts)
        end = self._normalize_time(end_ts)
        return self.redis.zrangebyscore(self.observed_key, start, end, withscores=True)

    # ------------------------------------------------------------------
    # Predicted series (API outputs)
    # ------------------------------------------------------------------
    def write_prediction(self, value: float, timestamp: float | int | datetime) -> None:
        ts = self._normalize_time(timestamp)
        if self.observed_exists(ts):
            return
        self.redis.zadd(self.predicted_key, {str(value).encode("utf-8"): ts})

    def read_predictions(self, start_ts: float | int | datetime, end_ts: float | int | datetime) -> List[tuple[bytes, float]]:
        start = self._normalize_time(start_ts)
        end = self._normalize_time(end_ts)
        return self.redis.zrangebyscore(self.predicted_key, start, end, withscores=True)

    # ------------------------------------------------------------------
    # Time-series bootstrap (observed only)
    # ------------------------------------------------------------------
    def bootstrap(self, df: pd.DataFrame, cnt_column: str = "cnt", date_column: str = "dteday", hour_column: str = "hr") -> None:
        self.redis.delete(self.observed_key)
        self.redis.delete(self.predicted_key)
        self.redis.delete(self.cache_key)
        for _, row in df.iterrows():
            dt = datetime.fromisoformat(str(row[date_column]))
            ts = datetime(dt.year, dt.month, dt.day, int(row[hour_column])).timestamp()
            value = float(row[cnt_column])
            cnt_transformed = float(value ** 0.5)
            self.write_observed(cnt_transformed, ts)

    # ------------------------------------------------------------------
    # History retrieval
    # ------------------------------------------------------------------
    def read_history(self, end_timestamp: float | int | datetime, window_hours: int | None = None) -> pd.DataFrame:
        end_time = self._normalize_time(end_timestamp)
        if window_hours is None:
            start_ts = float("-inf")
        else:
            start_ts = end_time - (window_hours * 3600)

        observed = self.read_observed(start_ts, end_time)
        predicted = self.read_predictions(start_ts, end_time)

        combined: List[tuple[float, float]] = []
        combined.extend(observed)
        combined.extend(predicted)
        if not combined:
            return pd.DataFrame(columns=["cnt_transformed", "timestamp"])

        rows: List[dict] = []
        seen_scores = set()
        for member_bytes, score in sorted(combined, key=lambda x: x[1]):
            if score in seen_scores:
                continue
            seen_scores.add(score)
            try:
                value = float(member_bytes.decode("utf-8"))
            except ValueError:
                continue
            rows.append({"cnt_transformed": value, "timestamp": datetime.fromtimestamp(score)})

        df = pd.DataFrame(rows)
        return df.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Prediction cache
    # ------------------------------------------------------------------
    def get_cached_prediction(self, cache_key: str):
        return self.redis.hget(self.cache_key, cache_key)

    def store_cached_prediction(self, cache_key: str, payload: bytes) -> None:
        self.redis.hset(self.cache_key, cache_key, payload)
