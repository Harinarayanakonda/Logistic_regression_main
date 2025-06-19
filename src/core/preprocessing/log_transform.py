import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Dict, Union, Any
import logging
from scipy import stats
import traceback
import warnings
from dataclasses import dataclass
from enum import Enum, auto

class TransformationMethod(Enum):
    LOG = auto()
    BOX_COX = auto()
    YEO_JOHNSON = auto()

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
    """
    Production-ready numeric feature transformer with comprehensive error handling,
    logging, and validation. Supports log, Box-Cox, and Yeo-Johnson transformations.
    
    Features:
    - Strict input validation
    - Comprehensive logging
    - Detailed error handling
    - Skewness analysis
    - Multiple zero-handling strategies
    - Memory efficiency
    - Thread safety considerations
    - Type hints for better IDE support
    
    Example Usage:
    --------------
    >>> transformer = LogTransformer(
    ...     method='box-cox',
    ...     columns=['age', 'income'],
    ...     handle_zeros='offset',
    ...     offset=1.0,
    ...     min_skew_threshold=0.5
    ... )
    >>> transformed_data = transformer.fit_transform(df)
    """
    
    def __init__(self, 
                 method: str = "log", 
                 columns: Optional[List[str]] = None, 
                 handle_zeros: str = "offset", 
                 offset: float = 1.0,
                 min_skew_threshold: float = 0.5,
                 verbose: bool = False):
        """
        Initialize the transformer with validation.
        
        Args:
            method: Transformation method ('log', 'box-cox', 'yeo-johnson')
            columns: Specific columns to transform (None for all numeric)
            handle_zeros: How to handle zeros/negatives ('offset', 'ignore', or 'drop')
            offset: Value added to avoid log(0) or box-cox failure
            min_skew_threshold: Minimum absolute skewness to consider transformation
            verbose: Whether to enable verbose logging
            
        Raises:
            ValueError: If invalid parameters are provided
        """
        self._validate_init_params(method, handle_zeros, offset, min_skew_threshold)
        
        self.method = method.lower()
        self.columns = columns
        self.handle_zeros = handle_zeros.lower()
        self.offset = offset
        self.min_skew_threshold = min_skew_threshold
        self.verbose = verbose
        
        # Internal state
        self.lambdas_: Dict[str, Optional[float]] = {} 
        self.fitted_columns_: List[str] = []
        self.skewness_: Optional[pd.Series] = None
        self._is_fitted: bool = False
        self._logger = self._initialize_logger()
        
    def _validate_init_params(self, 
                            method: str, 
                            handle_zeros: str,
                            offset: float,
                            min_skew_threshold: float) -> None:
        """Validate all initialization parameters."""
        valid_methods = ['log', 'box-cox', 'yeo-johnson']
        if method.lower() not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {method}")
        
        valid_zero_handling = ['offset', 'ignore', 'drop']
        if handle_zeros.lower() not in valid_zero_handling:
            raise ValueError(f"handle_zeros must be one of {valid_zero_handling}, got {handle_zeros}")
            
        if offset < 0:
            raise ValueError(f"Offset must be >= 0, got {offset}")
            
        if min_skew_threshold < 0:
            raise ValueError(f"min_skew_threshold must be >= 0, got {min_skew_threshold}")

    def _initialize_logger(self) -> logging.Logger:
        """Initialize and configure logger."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        logger.info("LogTransformer initialized with method: %s", self.method)
        return logger

    def _validate_input_data(self, X: Any) -> pd.DataFrame:
        """Validate input data and convert to DataFrame if needed."""
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError("Input must be pandas DataFrame, Series, or numpy array")
            
        if isinstance(X, (pd.Series, np.ndarray)):
            X = pd.DataFrame(X)
            
        if X.empty:
            raise ValueError("Input data is empty")
            
        return X.copy()

    def _select_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select numeric columns for transformation."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            self._logger.warning("No numeric columns found in input data")
            return pd.DataFrame()
            
        if self.columns:
            # Only use specified columns that exist and are numeric
            valid_cols = [col for col in self.columns 
                         if col in numeric_cols and col in X.columns]
            if not valid_cols:
                self._logger.warning("No valid numeric columns from specified list")
            return X[valid_cols]
            
        return X[numeric_cols]

    def _calculate_skewness(self, X: pd.DataFrame) -> pd.Series:
        """Calculate skewness with robust error handling."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                skewness = X.skew(numeric_only=True).sort_values(ascending=False)
                self._logger.debug("Skewness calculation completed")
                return skewness
            except Exception as e:
                self._logger.error("Failed to calculate skewness: %s", str(e))
                raise ValueError("Skewness calculation failed") from e

    def _should_transform_column(self, col: str, data: pd.Series) -> bool:
        """Determine if a column should be transformed."""
        if data.empty or data.nunique() <= 1:
            self._logger.debug("Skipping column %s: empty or constant values", col)
            return False
            
        if self.skewness_ is not None and abs(self.skewness_[col]) < self.min_skew_threshold:
            self._logger.debug("Skipping column %s: low skewness (%.2f)", 
                             col, self.skewness_[col])
            return False
            
        return True

    def _handle_non_positive(self, data: pd.Series, col: str) -> Optional[pd.Series]:
        """Handle non-positive values based on configuration."""
        non_pos_mask = data <= 0
        
        if not non_pos_mask.any():
            return data
            
        if self.handle_zeros == 'offset':
            adjusted = data + self.offset
            if (adjusted <= 0).any():
                self._logger.warning("Column %s still has non-positive values after offset", col)
                return None
            return adjusted
            
        elif self.handle_zeros == 'ignore':
            self._logger.warning("Column %s has non-positive values being ignored", col)
            return data
            
        elif self.handle_zeros == 'drop':
            self._logger.info("Dropping %d non-positive values in column %s", 
                            non_pos_mask.sum(), col)
            return data[~non_pos_mask]

    def _apply_transformation(self, data: pd.Series, col: str) -> Optional[pd.Series]:
        """Apply the configured transformation to a single column."""
        try:
            if self.method == 'log':
                if self.handle_zeros == 'offset':
                    data = data + self.offset
                return np.log(data)
                
            elif self.method == 'box-cox':
                data = self._handle_non_positive(data, col)
                if data is None:
                    return None
                transformed, lmbda = stats.boxcox(data)
                self.lambdas_[col] = lmbda
                return transformed
                
            elif self.method == 'yeo-johnson':
                transformed, lmbda = stats.yeojohnson(data)
                self.lambdas_[col] = lmbda
                return transformed
                
        except Exception as e:
            self._logger.error("Error transforming column %s: %s", col, str(e))
            return None

    def fit(self, X: pd.DataFrame, y=None) -> 'LogTransformer':
        """Fit the transformer to the data."""
        try:
            X = self._validate_input_data(X)
            numeric_df = self._select_numeric_columns(X)
            
            if numeric_df.empty:
                self._logger.warning("No numeric columns available for transformation")
                return self
                
            self.skewness_ = self._calculate_skewness(numeric_df)
            self.fitted_columns_ = numeric_df.columns.tolist()
            
            for col in self.fitted_columns_:
                data = numeric_df[col].dropna()
                
                if not self._should_transform_column(col, data):
                    self.lambdas_[col] = None
                    continue
                    
                if self.method in ['box-cox', 'yeo-johnson']:
                    data = self._handle_non_positive(data, col)
                    if data is None:
                        self.lambdas_[col] = None
                        continue
                        
                self.lambdas_[col] = None  # Will be set during transform
                
            self._is_fitted = True
            self._logger.info("Successfully fitted transformer")
            
        except Exception as e:
            self._logger.error("Fitting failed: %s", str(e))
            raise
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation to new data."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transformation")
            
        try:
            X = self._validate_input_data(X)
            X_transformed = X.copy()
            
            for col in self.fitted_columns_:
                if col not in X_transformed.columns:
                    self._logger.warning("Column %s not found in input data", col)
                    continue
                    
                data = X_transformed[col]
                valid_mask = data.notna()
                
                if not valid_mask.any():
                    self._logger.debug("Column %s has no valid values", col)
                    continue
                    
                transformed_values = self._apply_transformation(data[valid_mask], col)
                
                if transformed_values is not None:
                    X_transformed.loc[valid_mask, col] = transformed_values
                else:
                    self._logger.warning("Transformation failed for column %s - keeping original", col)
                    
            return X_transformed
            
        except Exception as e:
            self._logger.error("Transformation failed: %s", str(e))
            raise

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one operation."""
        return self.fit(X, y).transform(X)

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