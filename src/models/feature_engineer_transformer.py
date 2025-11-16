"""Scikit-learn transformer with Redis-backed stateful feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Tuple, Any

from sklearn.base import BaseEstimator, TransformerMixin

from src.config import ConfigLoader
from src.data.feature_engineering import FeatureEngineer
from src.utils.redis_manager import RedisManager


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """Stateful transformer that rebuilds lag/rolling features using Redis history."""

    def __init__(
        self,
        config: Optional[ConfigLoader] = None,
        history_window: Optional[int] = None,
        use_redis: bool = False,
    ) -> None:
        self.config = config or ConfigLoader()
        self.history_window = history_window
        self.use_redis = use_redis
        self.redis_manager: Optional[RedisManager] = None
        self._feature_engineer: Optional[FeatureEngineer] = None
        self._historical_avg_map: Dict[Tuple[int, int], float] = {}
        self._target_col = self.config.get("data.target_col", "cnt")
        self._base_year = int(self.config.get("features.base_year", 2011))
        self._fitted = False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _build_timestamp(self, df: pd.DataFrame) -> pd.Series:
        """Build timestamp from available columns.
        
        Tries multiple strategies:
        1. Use dteday + hr if available
        2. Use yr, mnth, hr if available (assumes day=1)
        3. Use existing timestamp column if available
        4. Returns a dummy timestamp series if none available
        """
        # Strategy 1: Use dteday + hr
        if "dteday" in df.columns and "hr" in df.columns:
            dt = pd.to_datetime(df["dteday"])
            return dt + pd.to_timedelta(df["hr"].astype(int), unit="h")
        
        # Strategy 2: Use existing timestamp column
        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"])
        
        # Strategy 3: Try to build from yr, mnth, hr (assume day=1)
        if "yr" in df.columns and "mnth" in df.columns and "hr" in df.columns:
            years = df["yr"].astype(int) + self._base_year
            months = df["mnth"].astype(int)
            hours = df["hr"].astype(int)
            # Create timestamp assuming day=1 of each month
            timestamps = pd.to_datetime({
                'year': years,
                'month': months,
                'day': 1,
                'hour': hours
            })
            return timestamps
        
        # Strategy 4: Return dummy timestamps (sequential hours from a base date)
        # This allows the transformer to work even without temporal info
        n_samples = len(df)
        base_date = pd.Timestamp(f'{self._base_year}-01-01')
        return pd.date_range(base_date, periods=n_samples, freq='H')

    def _apply_historical_average(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._historical_avg_map:
            return df
        hr = df.get("hr", pd.Series([], dtype=int)).fillna(0).astype(int)
        weekday = df.get("weekday", pd.Series([], dtype=int)).fillna(0).astype(int)
        averages = [self._historical_avg_map.get((h, w), 0.0) for h, w in zip(hr, weekday)]
        df["cnt_historical_avg_raw"] = averages
        df["cnt_vs_historical"] = df.get("cnt_transformed_lag_1h", pd.Series(0.0, index=df.index)) - df["cnt_historical_avg_raw"]
        return df

    # ------------------------------------------------------------------
    # scikit-learn interface
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self._feature_engineer = FeatureEngineer(self.config)
        df = X.copy()
        if y is not None:
            df[self._target_col] = y
        df_eng = self._feature_engineer.engineer_features(df)
        df_eng = self._feature_engineer.create_historical_context_features(df_eng, df_eng)
        if self._target_col in df_eng:
            df_eng["cnt_transformed"] = np.sqrt(df_eng[self._target_col].clip(lower=0))
        self._historical_avg_map = (
            df_eng.groupby(["hr", "weekday"]).agg({"cnt_transformed": "mean"}).iloc[:, 0].to_dict()
            if "cnt_transformed" in df_eng
            else {}
        )
        self._fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        model_identifier: Optional[str] = None,
        context_key: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineerTransformer must be fitted before calling transform().")
        if self._feature_engineer is None:
            raise RuntimeError("Internal FeatureEngineer not initialised.")

        self._ensure_redis_manager()

        df_in = X.copy()
        df_in = df_in.reset_index(drop=True)

        if "dteday" in df_in.columns:
            df_in["dteday"] = pd.to_datetime(df_in["dteday"])
            if "yr" not in df_in.columns:
                df_in["yr"] = (df_in["dteday"].dt.year - self._base_year).astype(int)
            else:
                df_in["yr"] = pd.to_numeric(df_in["yr"], errors="coerce").fillna(
                    (df_in["dteday"].dt.year - self._base_year).astype(int)
                )
            if "mnth" not in df_in.columns:
                df_in["mnth"] = df_in["dteday"].dt.month.astype(int)
            else:
                df_in["mnth"] = pd.to_numeric(df_in["mnth"], errors="coerce").fillna(
                    df_in["dteday"].dt.month.astype(int)
                )

        base_numeric_cols = [
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
        ]
        for col in base_numeric_cols:
            if col in df_in.columns:
                df_in[col] = pd.to_numeric(df_in[col], errors="coerce")
        numeric_base_cols = [
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
        ]
        for col in numeric_base_cols:
            if col in df_in.columns:
                df_in[col] = pd.to_numeric(df_in[col], errors="coerce")

        timestamps = self._build_timestamp(df_in)
        df_in["timestamp"] = pd.to_datetime(timestamps)
        df_in["cnt"] = np.nan
        df_in["cnt_transformed"] = np.nan

        if self.use_redis and self.redis_manager:
            end_timestamp = df_in["timestamp"].iloc[-1]
            history_df = self.redis_manager.read_history(end_timestamp, self.history_window)
        else:
            history_df = pd.DataFrame(columns=["cnt_transformed", "timestamp"])
        if not history_df.empty:
            history_df = history_df.copy()
            history_df["cnt"] = np.square(history_df["cnt_transformed"])
            history_df["timestamp"] = history_df["timestamp"]
            for col in df_in.columns:
                if col not in history_df.columns:
                    history_df[col] = np.nan
            history_df = history_df[df_in.columns]
        else:
            history_df = pd.DataFrame(columns=df_in.columns)

        combined = pd.concat([history_df, df_in], ignore_index=True)
        combined = combined.sort_values("timestamp")
        numeric_cols = [
            "cnt",
            "cnt_transformed",
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
        ]
        for col in numeric_cols:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce").astype(float)

        sqrt_cnt = np.sqrt(combined["cnt"].clip(lower=0).astype(float))
        combined["cnt_transformed"] = combined["cnt_transformed"].where(~combined["cnt_transformed"].isna(), sqrt_cnt)

        engineered = self._feature_engineer.engineer_features(combined)
        engineered = self._apply_historical_average(engineered)
        result = engineered.tail(len(df_in))
        result.index = df_in.index
        base_cols = [col for col in df_in.columns if col not in {"timestamp", "cnt", "cnt_transformed"}]
        for col in base_cols:
            result[col] = df_in[col]
        # Remove helper columns that should not reach downstream preprocessing/model steps
        result = result.drop(columns=["dteday", "cnt", "cnt_transformed", "timestamp"], errors="ignore")
        
        # Handle NaNs: fill with 0 for lag/rolling features, forward fill for others, then fill remaining with 0
        # This ensures the transformer output is always valid for sklearn models
        lag_rolling_cols = [col for col in result.columns if any(x in col for x in ['lag', 'rolling', 'pct_change'])]
        for col in lag_rolling_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
        
        # Fill remaining NaNs with 0 (for any other features that might have NaNs)
        result = result.fillna(0.0)
        
        return result

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # Redis connections are not pickleable; recreate lazily after loading
        state["redis_manager"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self.redis_manager = None
        self.use_redis = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_redis_manager(self) -> None:
        if not self.use_redis:
            return
        if self.redis_manager is None:
            self.redis_manager = RedisManager()
