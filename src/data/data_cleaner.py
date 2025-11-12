"""
Data cleaning utilities.
Handles data cleaning operations including type conversion, null handling, and outlier removal.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations.
    
    This class encapsulates all data cleaning logic, making it reusable
    and testable. Follows single responsibility principle.
    """
    
    def __init__(self, config):
        """
        Initialize data cleaner.
        
        Args:
            config: ConfigLoader instance for accessing configuration
        """
        self.config = config
        self.exclude_cols = config.get("data.exclude_cols", [])
        self.base_year = int(config.get("features.base_year", 2011))
        logger.info("DataCleaner initialized")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.
        
        This method orchestrates all cleaning steps:
        1. Convert data types
        2. Handle null values
        3. Remove problematic columns
        4. Validate and correct ranges (corrects invalid values, preserves all rows)
        5. Validate data integrity
        
        Args:
            df: Raw DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
        
        df_clean = df.copy()
        
        # Step 1: Convert data types
        df_clean = self._convert_dtypes(df_clean)
        
        # Step 2: Handle null values
        df_clean = self._handle_nulls(df_clean)
        
        # Step 3: Remove problematic columns
        df_clean = self._remove_problematic_columns(df_clean)
        
        # Step 3b: Recompute year index from dteday to support future years
        if 'dteday' in df_clean.columns:
            dteday_series = df_clean['dteday']
            df_clean['yr'] = (dteday_series.dt.year - self.base_year).astype(int)
            df_clean['mnth'] = dteday_series.dt.month.astype(int)
        
        # Step 4: Validate and filter ranges (NEW)
        df_clean = self._validate_and_filter_ranges(df_clean)

        # Step 4b: Ensure casual + registered = cnt relationship
        df_clean = self._ensure_cnt_consistency(df_clean)
        
        # Step 5: Validate
        self._validate_cleaned_data(df_clean)
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        
        return df_clean
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        logger.debug("Converting data types...")
        
        # Convert date column if present
        if 'dteday' in df.columns:
            df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Ensure numeric columns are numeric
        numeric_cols = ['season', 'yr', 'mnth', 'hr', 'weekday', 'holiday',
                        'workingday', 'weathersit', 'temp', 'atemp', 'hum',
                        'windspeed', 'casual', 'registered', 'cnt']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle null values.
        
        Strategy:
        - For temporal data: forward fill (carry last known value)
        - For other numeric: forward fill then backward fill
        """
        logger.debug(f"Handling null values. Initial nulls: {df.isnull().sum().sum()}")
        
        # Forward fill temporal data
        temporal_cols = ['temp', 'atemp', 'hum', 'windspeed', 'weathersit']
        for col in temporal_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # For remaining nulls, use forward fill then backward fill
        df = df.ffill().bfill()
        
        # Final check: if any nulls remain, drop those rows
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with remaining nulls")
        
        logger.debug(f"Final nulls: {df.isnull().sum().sum()}")
        
        return df
    
    def _remove_problematic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that are problematic or cause data leakage.
        
        Columns to remove:
        - instant: Row index (not informative)
        - Columns specified in config exclude_cols (after feature engineering)
        """
        logger.debug("Removing problematic columns...")
        
        # Always remove instant if present
        cols_to_remove = ['instant']
        
        # Remove mixed_type_col if present (common issue)
        if 'mixed_type_col' in df.columns:
            cols_to_remove.append('mixed_type_col')
        
        # Remove columns that are in exclude list and exist
        for col in self.exclude_cols:
            if col in df.columns and col not in ['cnt', 'casual', 'registered', 'dteday']:
                # Don't remove target columns at cleaning stage
                cols_to_remove.append(col)
        
        # Remove duplicates
        cols_to_remove = list(set(cols_to_remove))
        
        existing_cols_to_remove = [col for col in cols_to_remove if col in df.columns]
        
        if existing_cols_to_remove:
            logger.debug(f"Removing columns: {existing_cols_to_remove}")
            df = df.drop(columns=existing_cols_to_remove)
        
        return df
    
    def _validate_and_filter_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and correct values that are outside expected ranges.
        
        This method CORRECTS (not filters) values outside expected ranges by:
        - Re-normalizing values using original formulas when possible
        - Clipping to valid range boundaries for clearly invalid values
        - Preserving all rows while ensuring data quality
        
        Expected ranges and normalization formulas:
        - temp: [0, 1] (normalized: (t+8)/47 where t in [-8, 39]°C)
          Reverse: t = normalized * 47 - 8
        - atemp: [0, 1] (normalized: (t+16)/66 where t in [-16, 50]°C)
          Reverse: t = normalized * 66 - 16
        - hum: [0, 1] (normalized: humidity/100 where humidity in [0, 100]%)
          Reverse: humidity = normalized * 100
        - windspeed: [0, 1] (normalized: speed/67 where speed in [0, 67])
          Reverse: speed = normalized * 67
        
        Args:
            df: DataFrame to validate and correct
            
        Returns:
            DataFrame with invalid values corrected (all rows preserved)
        """
        logger.debug("Validating and correcting values outside expected ranges...")
        
        initial_len = len(df)
        
        # Define expected ranges and normalization formulas
        range_checks = {
            'temp': {
                'min_norm': 0.0, 'max_norm': 1.0,
                'min_original': -8.0, 'max_original': 39.0,
                'reverse_formula': lambda x: x * 47 - 8,  # t = normalized * 47 - 8
                'normalize_formula': lambda t: (t + 8) / 47,  # normalized = (t + 8) / 47
                'correction_method': 'renormalize'  # Try to re-normalize if original makes sense
            },
            'atemp': {
                'min_norm': 0.0, 'max_norm': 1.0,
                'min_original': -16.0, 'max_original': 50.0,
                'reverse_formula': lambda x: x * 66 - 16,  # t = normalized * 66 - 16
                'normalize_formula': lambda t: (t + 16) / 66,  # normalized = (t + 16) / 66
                'correction_method': 'renormalize'
            },
            'hum': {
                'min_norm': 0.0, 'max_norm': 1.0,
                'min_original': 0.0, 'max_original': 100.0,
                'reverse_formula': lambda x: x * 100,  # humidity = normalized * 100
                'normalize_formula': lambda h: h / 100,  # normalized = humidity / 100
                'correction_method': 'renormalize'
            },
            'windspeed': {
                'min_norm': 0.0, 'max_norm': 1.0,
                'min_original': 0.0, 'max_original': 67.0,
                'reverse_formula': lambda x: x * 67,  # speed = normalized * 67
                'normalize_formula': lambda s: s / 67,  # normalized = speed / 67
                'correction_method': 'renormalize'
            },
            # Discrete/categorical features: just clip
            'hr': (0, 23, 'clip'),
            'weekday': (0, 6, 'clip'),
            'season': (1, 4, 'clip'),
            'mnth': (1, 12, 'clip'),
            'weathersit': (1, 4, 'clip'),
            'holiday': (0, 1, 'clip'),
            'workingday': (0, 1, 'clip'),
        }
        
        # Track statistics
        corrections_by_column = {}
        df_corrected = df.copy()
        
        for col, check_info in range_checks.items():
            if col not in df_corrected.columns:
                continue
            
            # Handle continuous features with re-normalization capability
            if isinstance(check_info, dict) and check_info.get('correction_method') == 'renormalize':
                min_norm = check_info['min_norm']
                max_norm = check_info['max_norm']
                min_orig = check_info['min_original']
                max_orig = check_info['max_original']
                reverse_formula = check_info['reverse_formula']
                normalize_formula = check_info['normalize_formula']
                
                # Check for values outside normalized range
                out_of_range = (df_corrected[col] < min_norm) | (df_corrected[col] > max_norm)
                n_invalid = out_of_range.sum()
                
                if n_invalid > 0:
                    corrections_by_column[col] = n_invalid
                    invalid_values = df_corrected.loc[out_of_range, col]
                    invalid_examples = invalid_values.head(5).tolist()
                    
                    # Strategy: Try to re-normalize if original value makes sense
                    # Otherwise, clip to valid range
                    clipped_to_min = 0
                    clipped_to_max = 0
                    renormalized_count = 0
                    
                    for idx in invalid_values.index:
                        norm_val = df_corrected.loc[idx, col]
                        
                        # Try to reverse-normalize to get original value
                        try:
                            orig_val = reverse_formula(norm_val)
                            
                            # Check if original value is within reasonable bounds
                            # Allow some tolerance beyond expected range (e.g., ±10%)
                            tolerance = (max_orig - min_orig) * 0.1  # 10% tolerance
                            reasonable_min = min_orig - tolerance
                            reasonable_max = max_orig + tolerance
                            
                            if reasonable_min <= orig_val <= reasonable_max:
                                # Re-normalize with the original value (clipped to valid range)
                                orig_val_clipped = np.clip(orig_val, min_orig, max_orig)
                                new_norm_val = normalize_formula(orig_val_clipped)
                                df_corrected.loc[idx, col] = new_norm_val
                                renormalized_count += 1
                            else:
                                # Original value is clearly invalid, clip normalized value
                                if norm_val < min_norm:
                                    df_corrected.loc[idx, col] = min_norm
                                    clipped_to_min += 1
                                else:
                                    df_corrected.loc[idx, col] = max_norm
                                    clipped_to_max += 1
                        except Exception:
                            # If reverse formula fails, just clip
                            if norm_val < min_norm:
                                df_corrected.loc[idx, col] = min_norm
                                clipped_to_min += 1
                            else:
                                df_corrected.loc[idx, col] = max_norm
                                clipped_to_max += 1
                    
                    logger.warning(
                        f"Found {n_invalid} values outside range [{min_norm}, {max_norm}] "
                        f"in column '{col}'. Examples: {invalid_examples}. "
                        f"Corrected: {renormalized_count} re-normalized, "
                        f"{clipped_to_min} clipped to min, {clipped_to_max} clipped to max."
                    )
            
            # Handle discrete/categorical features: just clip
            elif isinstance(check_info, tuple) and len(check_info) == 3:
                min_val, max_val, method = check_info
                
                out_of_range = (df_corrected[col] < min_val) | (df_corrected[col] > max_val)
                n_invalid = out_of_range.sum()
                
                if n_invalid > 0:
                    corrections_by_column[col] = n_invalid
                    invalid_values = df_corrected.loc[out_of_range, col]
                    invalid_examples = invalid_values.head(5).tolist()
                    
                    # Clip to valid range
                    df_corrected[col] = df_corrected[col].clip(lower=min_val, upper=max_val)
                    
                    clipped_to_min = ((df_corrected[col] == min_val) & out_of_range).sum()
                    clipped_to_max = ((df_corrected[col] == max_val) & out_of_range).sum()
                    
                    logger.warning(
                        f"Found {n_invalid} values outside range [{min_val}, {max_val}] "
                        f"in column '{col}'. Examples: {invalid_examples}. "
                        f"Corrected by clipping: {clipped_to_min} to min, {clipped_to_max} to max."
                    )
        
        n_corrected = sum(corrections_by_column.values())
        
        if n_corrected > 0:
            logger.warning(
                f"Corrected {n_corrected} invalid values across columns. "
                f"All {initial_len} rows preserved. "
                f"Corrections by column: {corrections_by_column}"
            )
            logger.info(f"Rows preserved: {len(df_corrected)} (100% of original)")
        else:
            logger.debug("All values are within expected ranges")
        
        return df_corrected

    def _ensure_cnt_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that casual + registered == cnt for every row.

        If discrepancies are found, cnt is recomputed from the sum of casual and registered.
        Missing values are imputed when possible to preserve rows.
        """
        required_cols = {"casual", "registered", "cnt"}
        if not required_cols.issubset(df.columns):
            logger.warning(
                "Skipping cnt consistency enforcement because required columns are missing: %s",
                required_cols - set(df.columns),
            )
            return df

        df_consistent = df.copy()

        for col in required_cols:
            df_consistent[col] = pd.to_numeric(df_consistent[col], errors="coerce")

        # Attempt to impute missing values using available columns
        mask_cnt_missing = df_consistent["cnt"].isna() & df_consistent["casual"].notna() & df_consistent["registered"].notna()
        df_consistent.loc[mask_cnt_missing, "cnt"] = (
            df_consistent.loc[mask_cnt_missing, "casual"] + df_consistent.loc[mask_cnt_missing, "registered"]
        )

        mask_casual_missing = (
            df_consistent["casual"].isna() & df_consistent["cnt"].notna() & df_consistent["registered"].notna()
        )
        df_consistent.loc[mask_casual_missing, "casual"] = (
            df_consistent.loc[mask_casual_missing, "cnt"] - df_consistent.loc[mask_casual_missing, "registered"]
        )

        mask_registered_missing = (
            df_consistent["registered"].isna() & df_consistent["cnt"].notna() & df_consistent["casual"].notna()
        )
        df_consistent.loc[mask_registered_missing, "registered"] = (
            df_consistent.loc[mask_registered_missing, "cnt"] - df_consistent.loc[mask_registered_missing, "casual"]
        )

        values = df_consistent[["casual", "registered", "cnt"]].to_numpy(dtype=float)

        row_idx = 0
        for casual_val, registered_val, cnt_val in values:
            if pd.isna(casual_val) or pd.isna(registered_val) or pd.isna(cnt_val):
                row_idx += 1
                continue

            sum_cr = casual_val + registered_val

            # Candidate adjustments (value, new_casual, new_registered, new_cnt)
            candidates = []

            # Adjust cnt to match sum of casual + registered
            candidates.append(
                (
                    abs(sum_cr - cnt_val),
                    casual_val,
                    registered_val,
                    sum_cr,
                )
            )

            # Adjust casual to match cnt - registered (if non-negative)
            casual_adjusted = cnt_val - registered_val
            if casual_adjusted >= 0:
                candidates.append(
                    (
                        abs(casual_adjusted - casual_val),
                        casual_adjusted,
                        registered_val,
                        cnt_val,
                    )
                )

            # Adjust registered to match cnt - casual (if non-negative)
            registered_adjusted = cnt_val - casual_val
            if registered_adjusted >= 0:
                candidates.append(
                    (
                        abs(registered_adjusted - registered_val),
                        casual_val,
                        registered_adjusted,
                        cnt_val,
                    )
                )

            # Select candidate with minimal absolute adjustment; in ties prefer keeping cnt unchanged
            candidates.sort(key=lambda item: (item[0], max(item[1], item[2], item[3])))
            _, new_casual, new_registered, new_cnt = candidates[0]

            values[row_idx, 0] = max(0, new_casual)
            values[row_idx, 1] = max(0, new_registered)
            values[row_idx, 2] = max(0, new_cnt)
            row_idx += 1

        df_consistent["casual"] = np.round(values[:, 0]).astype(int)
        df_consistent["registered"] = np.round(values[:, 1]).astype(int)
        df_consistent["cnt"] = np.round(values[:, 2]).astype(int)

        mismatches = (df_consistent["casual"] + df_consistent["registered"] - df_consistent["cnt"]).abs()
        if mismatches.any():
            logger.error("Failed to enforce casual + registered = cnt relationship on some rows.")
            raise ValueError("Unable to enforce cnt consistency across the dataset.")

        logger.debug("Enforced casual + registered = cnt consistency across dataset.")
        return df_consistent
    
    def _validate_cleaned_data(self, df: pd.DataFrame) -> None:
        """Validate that cleaned data meets quality standards."""
        # Check for nulls
        if df.isnull().sum().sum() > 0:
            raise ValueError("Cleaned data still contains null values")
        
        # Check for target column
        target_col = self.config.get("data.target_col", "cnt")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in cleaned data")
        
        # Check for reasonable number of rows
        if len(df) < 100:
            raise ValueError(f"Too few rows in cleaned data: {len(df)}")

        # Validate the relationship casual + registered = cnt
        cols = {"casual", "registered", "cnt"}
        if cols.issubset(df.columns):
            mismatches = (df["casual"] + df["registered"] - df["cnt"]).abs()
            if mismatches.any():
                raise ValueError("Cleaned data violates the casual + registered = cnt relationship.")
        
        logger.debug("Data validation passed")

