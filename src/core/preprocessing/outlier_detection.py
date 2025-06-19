import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutlierHandlingMethod(Enum):
    FLAG = auto()
    REMOVE = auto()
    CLIP = auto()
    WINSORIZE = auto()

class DetectionMethod(Enum):
    IQR = auto()
    ZSCORE = auto()
    MAHALANOBIS = auto()

@dataclass
class OutlierDetectionConfig:
    method: DetectionMethod = DetectionMethod.IQR
    factor: Union[float, Dict[str, float]] = 1.5
    handling: OutlierHandlingMethod = OutlierHandlingMethod.FLAG
    min_non_null_values: int = 10  # Minimum non-null values required for detection
    robust_zscore: bool = True  # Use median/MAD for zscore instead of mean/std
    ignore_constant_columns: bool = True # Skip columns with no variability

    # Removed the invalid `self.method=method` line from here.
    # Dataclass fields are defined directly, e.g., 'method: DetectionMethod = DetectionMethod.IQR'

class OutlierDetector:
    """
    Production-grade outlier detection and handling system with multiple methods.

    Features:
    - Multiple detection methods (IQR, Z-score, Mahalanobis)
    - Multiple handling strategies (flag, remove, clip, winsorize)
    - Column-specific thresholds
    - Comprehensive logging and validation
    - Detailed outlier reporting
    - Save/load functionality
    - Memory efficiency
    - Null value handling
    - Constant column detection
    """

    def __init__(self,
                 method: Union[str, DetectionMethod] = 'iqr', # Allow string for convenience
                 factor: Union[float, Dict[str, float]] = 1.5, # Added factor here
                 handling: Union[str, OutlierHandlingMethod] = 'flag', # Added handling for convenience
                 config: Optional[OutlierDetectionConfig] = None):
        
        # If a pre-configured config object is provided, use it.
        # Otherwise, build one from the provided arguments.
        if config is None:
            # Map string methods to Enum members
            if isinstance(method, str):
                try:
                    method_enum = DetectionMethod[method.upper()]
                except KeyError:
                    raise ValueError(f"Invalid detection method string: '{method}'")
            else:
                method_enum = method

            if isinstance(handling, str):
                try:
                    handling_enum = OutlierHandlingMethod[handling.upper()]
                except KeyError:
                    raise ValueError(f"Invalid handling method string: '{handling}'")
            else:
                handling_enum = handling

            self.config = OutlierDetectionConfig(
                method=method_enum,
                factor=factor,
                handling=handling_enum # Pass handling to config
            )
        else:
            self.config = config

        self.lower_bounds: Dict[str, float] = {}
        self.upper_bounds: Dict[str, float] = {}
        self._validate_config()
        self._fitted = False
        logger.info("OutlierDetector initialized with config: %s", self.config)

    def _validate_config(self):
        """Validate configuration parameters."""
        if not isinstance(self.config.method, DetectionMethod):
            raise ValueError(f"Invalid detection method: {self.config.method}")

        if not isinstance(self.config.handling, OutlierHandlingMethod):
            raise ValueError(f"Invalid handling method: {self.config.handling}")

        if isinstance(self.config.factor, dict):
            for col, val in self.config.factor.items():
                if val <= 0:
                    raise ValueError(f"Factor must be positive for column '{col}', got {val}")
        elif self.config.factor <= 0:
            raise ValueError(f"Factor must be positive, got {self.config.factor}")

    def fit(self, X: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> 'OutlierDetector':
        """Learn outlier thresholds from training data."""
        self._validate_input_data(X)
        numeric_cols = self._get_valid_numeric_cols(X, numeric_cols)

        self.lower_bounds = {}
        self.upper_bounds = {}
        for col in numeric_cols:
            series = X[col]
            non_null_count = series.count()

            if non_null_count < self.config.min_non_null_values:
                logger.warning(f"Skipping column '{col}' with only {non_null_count} non-null values (min required: {self.config.min_non_null_values}).")
                continue

            if self.config.ignore_constant_columns and series.nunique() == 1:
                logger.info(f"Skipping constant column: {col}")
                continue

            col_factor = (
                self.config.factor.get(col, self.config.factor)
                if isinstance(self.config.factor, dict)
                else self.config.factor
            )

            if self.config.method == DetectionMethod.IQR:
                self._fit_iqr(series, col, col_factor)
            elif self.config.method == DetectionMethod.ZSCORE:
                self._fit_zscore(series, col, col_factor)
            else:
                raise NotImplementedError(f"Method {self.config.method} not implemented")

        self._fitted = True
        logger.info("Fitting completed with bounds for %d columns", len(self.lower_bounds))
        return self

    def _fit_iqr(self, series: pd.Series, col: str, factor: float):
        """Calculate IQR bounds for a column."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

        if IQR == 0:
            logger.info(f"Zero IQR for {col}, using quartiles directly (Q1={Q1}, Q3={Q3}).")
            self.lower_bounds[col] = Q1
            self.upper_bounds[col] = Q3
        else:
            self.lower_bounds[col] = Q1 - factor * IQR
            self.upper_bounds[col] = Q3 + factor * IQR

    def _fit_zscore(self, series: pd.Series, col: str, factor: float):
        """Calculate Z-score bounds for a column."""
        if self.config.robust_zscore:
            center = series.median()
            # Median Absolute Deviation (MAD) is typically multiplied by a constant (e.g., 1.4826 for normal distribution)
            # to make it comparable to standard deviation.
            scale = series.mad()
            if scale == 0: # Handle cases where MAD is zero for constant data
                 logger.info(f"Zero MAD for {col}, using mean/std for Z-score calculation.")
                 center = series.mean()
                 scale = series.std()
        else:
            center = series.mean()
            scale = series.std()

        if scale == 0:
            logger.info(f"Zero scale (std/MAD) for {col}, using center directly (value={center}).")
            self.lower_bounds[col] = center
            self.upper_bounds[col] = center
        else:
            self.lower_bounds[col] = center - factor * scale
            self.upper_bounds[col] = center + factor * scale

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier handling to new data."""
        self._validate_input_data(X)
        if not self._fitted:
            logger.warning("Detector not fitted. Returning original data.")
            return X.copy()

        X_copy = X.copy()
        numeric_cols = [col for col in self.lower_bounds.keys() if col in X_copy.columns]

        if not numeric_cols:
            logger.warning("No fitted numeric columns found in input data for transformation.")
            return X_copy

        outlier_flags = self._detect_outliers(X_copy, numeric_cols)

        if self.config.handling == OutlierHandlingMethod.FLAG:
            return self._handle_flag(X_copy, outlier_flags)
        elif self.config.handling == OutlierHandlingMethod.REMOVE:
            return self._handle_remove(X_copy, outlier_flags)
        elif self.config.handling == OutlierHandlingMethod.CLIP:
            return self._handle_clip(X_copy, numeric_cols)
        elif self.config.handling == OutlierHandlingMethod.WINSORIZE:
            return self._handle_winsorize(X_copy, numeric_cols)
        else:
            raise ValueError(f"Unknown handling method: {self.config.handling}")
    def _handle_flag(self, X: pd.DataFrame, outlier_flags: pd.DataFrame) -> pd.DataFrame:
            """Add a single outlier flag column (1 if any outlier exists in row, else 0)."""
            X_copy = X.copy()
            existing_flag_cols = [col for col in X_copy.columns if col.startswith('Outlier_Flag')]
            if existing_flag_cols:
                X_copy = X_copy.drop(columns=existing_flag_cols)
            X_copy['Outlier_Flag'] = outlier_flags.any(axis=1).astype(int)
            outlier_count = X_copy['Outlier_Flag'].sum()
            logger.info(f"Outliers detected in {outlier_count} rows ({outlier_count/len(X_copy):.2%})")
            return X_copy




    def _handle_remove(self, X: pd.DataFrame, outlier_flags: pd.DataFrame) -> pd.DataFrame:
        """Remove rows containing outliers."""
        initial_rows = X.shape[0]
        cleaned_df = X[~outlier_flags.any(axis=1)].copy() # Use .copy() to avoid SettingWithCopyWarning
        removed_rows = initial_rows - cleaned_df.shape[0]
        logger.info(f"Removed {removed_rows} rows due to outliers.")
        return cleaned_df

    def _handle_clip(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Clip values to the detected bounds."""
        X_copy = X.copy()
        for col in cols:
            # We don't necessarily need original min/max for logging clipped count.
            # Instead, we can compare values to bounds.
            clipped_lower_count = (X_copy[col] < self.lower_bounds[col]).sum()
            clipped_upper_count = (X_copy[col] > self.upper_bounds[col]).sum()
            
            X_copy[col] = X_copy[col].clip(
                lower=self.lower_bounds[col],
                upper=self.upper_bounds[col]
            )
            
            total_clipped_in_col = clipped_lower_count + clipped_upper_count
            if total_clipped_in_col > 0:
                logger.debug(f"Clipped {total_clipped_in_col} values in column '{col}'.")
        logger.info("Outliers clipped to bounds.")
        return X_copy

    def _handle_winsorize(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Winsorize values (clip and preserve outliers as separate columns)."""
        X_copy = X.copy()
        total_winsorized = 0
        
        # Remove any existing original_col_outlier or outlier_flag_col columns
        existing_winsorize_cols = [
            col for col in X_copy.columns 
            if col.startswith('Original_') and col.endswith('_Outlier') or col.startswith('Outlier_Flag_')
        ]
        if existing_winsorize_cols:
            X_copy = X_copy.drop(columns=existing_winsorize_cols)

        for col in cols:
            outlier_mask = (
                (X_copy[col] < self.lower_bounds[col]) |
                (X_copy[col] > self.upper_bounds[col])
            ) & X_copy[col].notna()

            # Save original outliers for values that are outside the bounds AND not NaN
            # Use .loc to avoid SettingWithCopyWarning
            original_outlier_col_name = f'Original_{col}_Outlier'
            # Ensure unique column name if already exists
            if original_outlier_col_name in X_copy.columns:
                i = 1
                while f'{original_outlier_col_name}_{i}' in X_copy.columns:
                    i += 1
                original_outlier_col_name = f'{original_outlier_col_name}_{i}'
            X_copy.loc[outlier_mask, original_outlier_col_name] = X_copy.loc[outlier_mask, col]
            
            # Clip values
            X_copy[col] = X_copy[col].clip(
                lower=self.lower_bounds[col],
                upper=self.upper_bounds[col]
            )
            
            # Add outlier flags
            flag_col_name = f'Outlier_Flag_{col}'
            if flag_col_name in X_copy.columns:
                i = 1
                while f'{flag_col_name}_{i}' in X_copy.columns:
                    i += 1
                flag_col_name = f'{flag_col_name}_{i}'
            X_copy[flag_col_name] = outlier_mask.astype(int)
            total_winsorized += outlier_mask.sum()

        # Add combined flag column
        any_flag_col_name = 'Outlier_Flag_Any'
        if any_flag_col_name in X_copy.columns:
            i = 1
            while f'{any_flag_col_name}_{i}' in X_copy.columns:
                i += 1
            any_flag_col_name = f'{any_flag_col_name}_{i}'
        X_copy[any_flag_col_name] = X_copy.filter(like='Outlier_Flag_').any(axis=1).astype(int)

        logger.info(f"Winsorized {total_winsorized} outlier values and added flags.")
        return X_copy

    def _detect_outliers(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Detect outliers across specified columns, returning a boolean DataFrame."""
        outlier_mask = pd.DataFrame(False, index=X.index, columns=cols)
        for col in cols:
            # Only check non-null values
            non_null_mask = X[col].notna()
            if non_null_mask.any():
                col_data = X.loc[non_null_mask, col]
                outlier_mask.loc[non_null_mask, col] = (
                    (col_data < self.lower_bounds[col]) |
                    (col_data > self.upper_bounds[col])
                )
        return outlier_mask

    def _validate_input_data(self, X: pd.DataFrame):
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")
        if X.empty:
            logger.warning("Input DataFrame is empty.")
            # For fit, an empty DataFrame might be a problem, but for transform,
            # it might just mean no data to process. Let's make it a warning here.
            # raise ValueError("Input DataFrame is empty") # Uncomment if you want to strictly enforce non-empty

    def _get_valid_numeric_cols(self, X: pd.DataFrame, cols: Optional[List[str]]) -> List[str]:
        """Get valid numeric columns with validation."""
        if cols is None:
            return X.select_dtypes(include=np.number).columns.tolist()

        valid_cols = []
        for col in cols:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
            elif not pd.api.types.is_numeric_dtype(X[col]): # More robust check for numeric dtype
                logger.warning(f"Column '{col}' is not numeric. Skipping.")
            else:
                valid_cols.append(col)
        return valid_cols

    def fit_transform(self, X: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, numeric_cols).transform(X)

    def detect_outliers_table(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive outlier detection report."""
        self._validate_input_data(X)
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call 'fit' before generating detection table.")

        numeric_cols = [col for col in self.lower_bounds.keys() if col in X.columns]
        if not numeric_cols:
            logger.warning("No fitted numeric columns in input data to generate outlier table.")
            return pd.DataFrame(columns=[
                "Column", "Outliers Detected", "% of Values",
                "Lower Bound", "Upper Bound", "Non-Null Count"
            ])

        summary = []
        for col in numeric_cols:
            series = X[col]
            total_non_null = series.count()

            if total_non_null == 0:
                logger.debug(f"Column '{col}' has no non-null values. Skipping in summary.")
                continue

            non_null_mask = series.notna()
            if non_null_mask.any():
                col_data = series[non_null_mask]
                outliers = col_data[
                    (col_data < self.lower_bounds[col]) |
                    (col_data > self.upper_bounds[col])
                ]
                count_outliers = outliers.count()
                percent = round((count_outliers / total_non_null) * 100, 2) if total_non_null > 0 else 0.0

                summary.append({
                    "Column": col,
                    "Outliers Detected": count_outliers,
                    "% of Values": percent,
                    "Lower Bound": round(self.lower_bounds[col], 4),
                    "Upper Bound": round(self.upper_bounds[col], 4),
                    "Non-Null Count": total_non_null
                })
        
        if not summary:
            return pd.DataFrame(columns=[
                "Column", "Outliers Detected", "% of Values",
                "Lower Bound", "Upper Bound", "Non-Null Count"
            ])

        return pd.DataFrame(summary).sort_values("Outliers Detected", ascending=False)

    def get_outlier_rows(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get rows containing outliers with detailed information."""
        self._validate_input_data(X)
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call 'fit' before getting outlier rows.")

        X_copy = X.copy()
        numeric_cols = [col for col in self.lower_bounds.keys() if col in X_copy.columns]

        if not numeric_cols:
            logger.warning("No fitted numeric columns in input data to retrieve outlier rows.")
            return pd.DataFrame(columns=X.columns.tolist() + ['outlier_flag', 'outlier_columns'])

        outlier_flags = self._detect_outliers(X_copy, numeric_cols)

        # Add outlier information
        # Ensure 'outlier_flag' is not duplicated
        any_flag_col_name = 'outlier_flag'
        if any_flag_col_name in X_copy.columns:
            i = 1
            while f'{any_flag_col_name}_{i}' in X_copy.columns:
                i += 1
            any_flag_col_name = f'{any_flag_col_name}_{i}'
        X_copy[any_flag_col_name] = outlier_flags.any(axis=1).astype(int)

        # Ensure 'outlier_columns' is not duplicated
        outlier_cols_col_name = 'outlier_columns'
        if outlier_cols_col_name in X_copy.columns:
            i = 1
            while f'{outlier_cols_col_name}_{i}' in X_copy.columns:
                i += 1
            outlier_cols_col_name = f'{outlier_cols_col_name}_{i}'
        X_copy[outlier_cols_col_name] = outlier_flags.apply(
            lambda row: [col for col in numeric_cols if row[col]], axis=1)

        # Return only rows with outliers
        return X_copy[X_copy[any_flag_col_name] == 1].copy() # Use the dynamically generated column name


    def save(self, filepath: str):
        """Save detector state to file."""
        state = {
            'config': self.config,
            'lower_bounds': self.lower_bounds,
            'upper_bounds': self.upper_bounds,
            '_fitted': self._fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info("Saved detector state to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> 'OutlierDetector':
        """Load detector from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # When loading, we use the saved config to initialize the new instance
        detector = cls(config=state['config'])
        detector.lower_bounds = state['lower_bounds']
        detector.upper_bounds = state['upper_bounds']
        detector._fitted = state['_fitted']
        
        logger.info("Loaded detector from %s", filepath)
        return detector

    def __repr__(self):
        return (f"OutlierDetector(method={self.config.method.name}, " # Use .name for readable enum string
                f"handling={self.config.handling.name}, " # Use .name for readable enum string
                f"factor={self.config.factor}, "
                f"fitted={self._fitted}, "
                f"columns={len(self.lower_bounds)})")