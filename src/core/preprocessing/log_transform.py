# src/core/preprocessing/log_transform.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from typing import List, Optional, Dict, Any
import logging
from scipy import stats
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from scipy.stats.mstats import winsorize
from src.core.preprocessing.utils import separate_features_and_target, recombine_features_and_target

# Reuse this class in Streamlit: from src.core.preprocessing.log_transform import LogTransformer

class TransformationMethod(Enum):
    LOG = auto()
    BOX_COX = auto()
    YEO_JOHNSON = auto()
    SQRT = auto()
    CBRT = auto()
    INVERSE = auto()
    POWER = auto()
    WINSORIZING = auto()
    CLIPPING = auto()
    CONSTANT_ADDITION = auto()
    BINNING = auto()
    QUANTILE = auto()
    ZSCORE = auto()

class ZeroHandling(Enum):
    OFFSET = auto()
    IGNORE = auto()
    DROP = auto()

@dataclass
class TransformationResult:
    success: bool
    transformed_data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    skewness_report: Optional[pd.DataFrame] = None

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method: str = "log",
        columns: Optional[List[str]] = None,
        handle_zeros: str = "offset",
        offset: float = 1.0,
        min_skew_threshold: float = 0.5,
        verbose: bool = False,
        power_value: float = 2.0
    ):
        self._validate_init_params(method, handle_zeros, offset, min_skew_threshold)
        self.method = method.lower()
        self.columns = columns
        self.handle_zeros = handle_zeros.lower()
        self.offset = offset
        self.min_skew_threshold = min_skew_threshold
        self.verbose = verbose
        self.power_value = power_value
        self.lambdas_: Dict[str, Optional[float]] = {}
        self.fitted_columns_: List[str] = []
        self.skewness_: Optional[pd.Series] = None
        self._is_fitted: bool = False
        self._logger = self._initialize_logger()

    def _initialize_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.info("LogTransformer initialized with method: %s", self.method)
        return logger

    def _validate_init_params(self, method: str, handle_zeros: str, offset: float, min_skew_threshold: float) -> None:
        valid_methods = [m.name.lower() for m in TransformationMethod]
        if method.lower() not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {method}")
        valid_zero_handling = [z.name.lower() for z in ZeroHandling]
        if handle_zeros.lower() not in valid_zero_handling:
            raise ValueError(f"handle_zeros must be one of {valid_zero_handling}, got {handle_zeros}")
        if offset < 0 or min_skew_threshold < 0:
            raise ValueError("Offset and min_skew_threshold must be non-negative")

    def _validate_input_data(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            return X.to_frame()
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            return X.copy()
        else:
            raise TypeError("Input must be DataFrame, Series, or ndarray")

    def _select_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.columns:
            return X[[col for col in self.columns if col in numeric_cols]]
        return X[numeric_cols]

    def _calculate_skewness(self, X: pd.DataFrame) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return X.skew(numeric_only=True).sort_values(ascending=False)

    def _should_transform_column(self, col: str, data: pd.Series) -> bool:
        if data.empty or data.nunique() <= 1:
            return False
        if self.skewness_ is not None and abs(self.skewness_.get(col, 0)) < self.min_skew_threshold:
            return False
        return True

    def _handle_non_positive(self, data: pd.Series, col: str) -> Optional[pd.Series]:
        if self.handle_zeros == 'offset':
            data = data + self.offset
            return data if (data > 0).all() else None
        elif self.handle_zeros == 'ignore':
            return data
        elif self.handle_zeros == 'drop':
            return data[data > 0]
        return None

    def _apply_transformation(self, data: pd.Series, col: str) -> Optional[pd.Series]:
        try:
            if self.method == 'log':
                return np.log(data + self.offset) if self.handle_zeros == 'offset' else np.log(data)
            elif self.method == 'box-cox':
                data = self._handle_non_positive(data, col)
                if data is None: return None
                transformed, lmbda = stats.boxcox(data)
                self.lambdas_[col] = lmbda
                return transformed
            elif self.method == 'yeo-johnson':
                transformed, lmbda = stats.yeojohnson(data)
                self.lambdas_[col] = lmbda
                return transformed
            elif self.method == 'square-root':
                return np.sqrt(data)
            elif self.method == 'cube-root':
                return np.cbrt(data)
            elif self.method == 'inverse':
                return 1 / (data + self.offset)
            elif self.method == 'power':
                return np.power(data, self.power_value)
            elif self.method == 'winsorizing':
                return winsorize(data, limits=[0.01, 0.01])
            elif self.method == 'clipping':
                return np.clip(data, data.quantile(0.01), data.quantile(0.99))
            elif self.method == 'constant-addition':
                return data + self.offset
            elif self.method == 'data-binning':
                return pd.qcut(data, q=4, labels=False, duplicates='drop')
            elif self.method == 'quantile-transform':
                qt = QuantileTransformer(output_distribution="normal")
                return qt.fit_transform(data.values.reshape(-1, 1)).flatten()
            elif self.method == 'z-score':
                return (data - data.mean()) / data.std()
        except Exception as e:
            self._logger.error(f"Error transforming column {col}: {str(e)}")
            return None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._validate_input_data(X)
        numeric_df = self._select_numeric_columns(X)
        self.skewness_ = self._calculate_skewness(numeric_df)
        self.fitted_columns_ = numeric_df.columns.tolist()
        for col in self.fitted_columns_:
            data = numeric_df[col].dropna()
            if not self._should_transform_column(col, data):
                self.lambdas_[col] = None
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        X = self._validate_input_data(X)
        X_transformed = X.copy()

        for col in self.fitted_columns_:
            if col not in X_transformed.columns:
                continue

            data = X_transformed[col]
            valid_mask = data.notna()
            transformed_values = self._apply_transformation(data[valid_mask], col)

            if transformed_values is not None:
                X_transformed.loc[valid_mask, col] = transformed_values
            else:
                self._logger.warning(f"Transformation failed for column '{col}', keeping original values.")

        return X_transformed


    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformation on the data."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before inverse transformation")
            
        try:
            X = self._validate_input_data(X)
            X_inverted = X.copy()
            
            for col in self.fitted_columns_:
                if col not in X_inverted.columns:
                    continue
                    
                data = X_inverted[col]
                valid_mask = data.notna()
                
                if not valid_mask.any():
                    continue
                    
                if self.method == 'log':
                    inverted = np.exp(data[valid_mask])
                    if self.handle_zeros == 'offset':
                        inverted -= self.offset
                    X_inverted.loc[valid_mask, col] = inverted
                    
                elif self.method == 'box-cox' and col in self.lambdas_:
                    X_inverted.loc[valid_mask, col] = stats.inv_boxcox(
                        data[valid_mask], 
                        self.lambdas_[col]
                    )
                    
                elif self.method == 'yeo-johnson':
                    self._logger.warning("Inverse Yeo-Johnson not implemented - returning as-is")
                    
            return X_inverted
            
        except Exception as e:
            self._logger.error("Inverse transformation failed: %s", str(e))
            raise

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Get output feature names for transformation."""
        if input_features is None:
            return self.fitted_columns_
        return [f for f in input_features if f in self.fitted_columns_]

    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'method': self.method,
            'columns': self.columns,
            'handle_zeros': self.handle_zeros,
            'offset': self.offset,
            'min_skew_threshold': self.min_skew_threshold,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> 'LogTransformer':
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        self._validate_init_params(
            self.method, 
            self.handle_zeros,
            self.offset,
            self.min_skew_threshold
        )
        return self

    def get_skewness_report(self) -> Optional[pd.DataFrame]:
        """Get a report of skewness values for transformed columns."""
        if self.skewness_ is None:
            return None
            
        report = pd.DataFrame({
            'feature': self.skewness_.index,
            'skewness': self.skewness_.values,
            'transformed': [
                col in self.fitted_columns_ and self.lambdas_.get(col, None) is not None
                for col in self.skewness_.index
            ],
            'lambda': [
                self.lambdas_.get(col, np.nan) 
                for col in self.skewness_.index
            ]
        })
        
        return report.sort_values('skewness', key=abs, ascending=False)

    def __repr__(self) -> str:
        """Official string representation of the transformer."""
        return (f"LogTransformer(method='{self.method}', "
                f"columns={self.columns}, "
                f"handle_zeros='{self.handle_zeros}', "
                f"offset={self.offset}, "
                f"min_skew_threshold={self.min_skew_threshold})")