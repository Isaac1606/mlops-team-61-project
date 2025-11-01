"""
Feature engineering utilities.
Creates new features from existing data, including lags, rolling statistics, and interactions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering operations.
    
    This class creates derived features from raw data, including:
    - Temporal features (lags, rolling statistics)
    - Cyclical encodings (sin/cos transformations)
    - Interaction features
    - Advanced features (volatilities, momentum, etc.)
    
    All feature engineering respects temporal order to avoid data leakage.
    """
    
    def __init__(self, config):
        """
        Initialize feature engineer.
        
        Args:
            config: ConfigLoader instance for accessing configuration
        """
        self.config = config
        self.feature_config = config.get_section("features")
        logger.info("FeatureEngineer initialized")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Cleaned DataFrame
        
        Returns:
            DataFrame with engineered features added
        """
        logger.info(f"Starting feature engineering. Initial shape: {df.shape}")
        
        df_features = df.copy()
        
        # Ensure data is sorted by time (critical for lags)
        if 'timestamp' in df_features.columns or 'dteday' in df_features.columns:
            time_col = 'timestamp' if 'timestamp' in df_features.columns else 'dteday'
            df_features = df_features.sort_values(time_col).reset_index(drop=True)
            logger.debug(f"Sorted data by {time_col}")
        
        # Step 1: Transform target (for creating derived features)
        df_features = self._transform_target(df_features)
        
        # Step 2: Create lag features
        df_features = self._create_lag_features(df_features)
        
        # Step 3: Create rolling statistics
        df_features = self._create_rolling_features(df_features)
        
        # Step 4: Create cyclical encodings
        df_features = self._create_cyclical_features(df_features)
        
        # Step 5: Create interaction features
        df_features = self._create_interaction_features(df_features)
        
        # Step 6: Create advanced features
        if self.feature_config.get("advanced_features", {}).get("volatility", False):
            df_features = self._create_volatility_features(df_features)
        
        if self.feature_config.get("advanced_features", {}).get("momentum", False):
            df_features = self._create_momentum_features(df_features)
        
        if self.feature_config.get("advanced_features", {}).get("weather_interactions", False):
            df_features = self._create_weather_interaction_features(df_features)
        
        # Step 7: Create temporal indicators
        df_features = self._create_temporal_indicators(df_features)
        
        logger.info(f"Feature engineering complete. Final shape: {df_features.shape}")
        
        return df_features
    
    def _transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sqrt transformation to target (helps with heteroscedasticity)."""
        target_col = self.config.get("data.target_col", "cnt")
        
        if target_col in df.columns:
            df['cnt_transformed'] = np.sqrt(df[target_col])
            logger.debug("Created cnt_transformed feature")
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for the target variable."""
        lag_features = self.feature_config.get("lag_features", [1, 24, 48, 72, 168])
        
        if 'cnt_transformed' not in df.columns:
            logger.warning("cnt_transformed not found, skipping lag features")
            return df
        
        for lag in lag_features:
            col_name = f'cnt_transformed_lag_{lag}h'
            df[col_name] = df['cnt_transformed'].shift(lag)
            logger.debug(f"Created lag feature: {col_name}")
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics."""
        rolling_windows = self.feature_config.get("rolling_windows", [3, 24, 72])
        
        if 'cnt_transformed' not in df.columns:
            return df
        
        for window in rolling_windows:
            # Rolling mean
            col_name = f'cnt_transformed_roll_mean_{window}h'
            df[col_name] = df['cnt_transformed'].rolling(window=window, min_periods=1).mean()
            
            # Rolling std
            col_name = f'cnt_transformed_roll_std_{window}h'
            df[col_name] = df['cnt_transformed'].rolling(window=window, min_periods=1).std().fillna(0)
            
            logger.debug(f"Created rolling features for window {window}h")
        
        return df
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sin/cos encodings for cyclical features."""
        cyclical_features = self.feature_config.get("cyclical_features", ["hr", "mnth", "weekday"])
        
        for feature in cyclical_features:
            if feature not in df.columns:
                continue
            
            # Get max value for normalization
            max_val = df[feature].max()
            
            # Sin and cos transformations
            df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / max_val)
            df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / max_val)
            
            logger.debug(f"Created cyclical features for {feature}")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        interactions = self.feature_config.get("interactions", [])
        
        for feat1, feat2 in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                col_name = f'{feat1}_{feat2}'
                df[col_name] = df[feat1] * df[feat2]
                logger.debug(f"Created interaction feature: {col_name}")
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features (24h window)."""
        if 'cnt_transformed' not in df.columns:
            return df
        
        # 24h rolling coefficient of variation
        roll_mean_24h = df['cnt_transformed'].rolling(window=24, min_periods=1).mean()
        roll_std_24h = df['cnt_transformed'].rolling(window=24, min_periods=1).std().fillna(0)
        df['cnt_cv_24h'] = np.where(roll_mean_24h > 0, roll_std_24h / roll_mean_24h, 0)
        
        # Volatility (std of changes)
        df['cnt_volatility_24h'] = df['cnt_transformed'].diff().rolling(window=24, min_periods=1).std().fillna(0)
        
        logger.debug("Created volatility features")
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum/acceleration features."""
        if 'cnt_transformed' not in df.columns:
            return df
        
        # Acceleration (rate of change of rate of change)
        df['cnt_acceleration_1h'] = df['cnt_transformed'].diff().diff()
        df['cnt_acceleration_24h'] = df['cnt_transformed'].diff(24).diff()
        
        # Percentage change
        df['cnt_pct_change_1h'] = df['cnt_transformed'].pct_change().fillna(0)
        df['cnt_pct_change_24h'] = df['cnt_transformed'].pct_change(24).fillna(0)
        
        logger.debug("Created momentum features")
        
        return df
    
    def _create_weather_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-related interaction features."""
        # Temperature squared (non-linear effect)
        if 'temp' in df.columns:
            df['temp_squared'] = df['temp'] ** 2
        
        # Temperature × humidity
        if 'temp' in df.columns and 'hum' in df.columns:
            df['temp_hum_interaction'] = df['temp'] * df['hum']
        
        # Temperature × windspeed
        if 'temp' in df.columns and 'windspeed' in df.columns:
            df['temp_wind_interaction'] = df['temp'] * df['windspeed']
        
        # Perfect weather indicator
        if all(col in df.columns for col in ['temp', 'hum', 'windspeed', 'weathersit']):
            df['is_perfect_weather'] = (
                (df['temp'] > 0.5) &  # Warm
                (df['hum'] < 0.7) &   # Not too humid
                (df['windspeed'] < 0.5) &  # Not too windy
                (df['weathersit'] == 1)  # Clear
            ).astype(int)
        
        logger.debug("Created weather interaction features")
        
        return df
    
    def _create_temporal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal indicator features."""
        # Peak hour indicator (7-9am, 5-7pm)
        if 'hr' in df.columns:
            df['is_peak_hour'] = (
                df['hr'].isin([7, 8, 9, 17, 18, 19])
            ).astype(int)
            
            # Commute window
            df['is_commute_window'] = (
                df['hr'].isin([7, 8, 9, 17, 18])
            ).astype(int)
        
        # Hour × working day interaction
        if 'hr' in df.columns and 'workingday' in df.columns:
            df['hr_workingday'] = df['hr'] * df['workingday']
        
        logger.debug("Created temporal indicator features")
        
        return df
    
    def create_historical_context_features(
        self,
        train_df: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create historical context features using ONLY training data.
        
        This method prevents data leakage by calculating historical averages
        only on training data, then applying them to other splits.
        
        Args:
            train_df: Training DataFrame (used for calculating averages)
            df: DataFrame to apply features to (can be train, val, or test)
        
        Returns:
            DataFrame with historical context features added
        """
        logger.debug("Creating historical context features (no leakage)")
        
        df_features = df.copy()
        
        # Calculate historical averages from training data only
        if 'cnt_transformed' in train_df.columns and 'hr' in train_df.columns and 'weekday' in train_df.columns:
            historical_avg = (
                train_df
                .groupby(['hr', 'weekday'])['cnt_transformed']
                .mean()
                .to_dict()
            )
            
            # Apply to current dataframe
            def get_historical_avg(row):
                key = (row['hr'], row['weekday'])
                return historical_avg.get(key, 0.0)
            
            df_features['cnt_historical_avg_raw'] = df_features.apply(get_historical_avg, axis=1)
            
            # Compare current value vs historical (using lag to avoid leakage)
            if 'cnt_transformed_lag_1h' in df_features.columns:
                df_features['cnt_vs_historical'] = (
                    df_features['cnt_transformed_lag_1h'] - df_features['cnt_historical_avg_raw']
                )
        
        logger.debug("Historical context features created")
        
        return df_features

