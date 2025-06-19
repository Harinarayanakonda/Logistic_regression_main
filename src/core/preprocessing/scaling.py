import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Dict, Union, Any
import logging
from enum import Enum, auto
import warnings
from dataclasses import dataclass

class ScalingMethod(Enum):
    STANDARD = auto()
    MINMAX = auto()
    ROBUST = auto()

@dataclass
class ScalingResult:
    success: bool
    scaled_data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    warnings: List[str] = None

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Production-ready feature scaler with comprehensive error handling and logging.
    Supports multiple scaling methods with robust data validation.
    
    Features:
    - StandardScaler (Z-score normalization)
    - MinMaxScaler (normalization to [0,1] range)
    - RobustScaler (outlier-resistant scaling)
    - Detailed logging and error handling
    - Column-wise scaling control
    - Memory efficiency
    - Thread safety considerations
    
    Example Usage:
    --------------
    >>> scaler = FeatureScaler(
    ...     method='standard',
    ...     columns=['age', 'income'],
    ...     verbose=True
    ... )
    >>> scaled_data = scaler.fit_transform(df)
    """
    
    def __init__(self, 
                 method: str = 'standard',
                 columns: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Initialize the feature scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', or 'robust')
            columns: Specific columns to scale (None for all numeric)
            verbose: Whether to enable verbose logging
            
        Raises:
            ValueError: If invalid parameters are provided
        """
        self._validate_init_params(method)
        
        self.method = method.lower()
        self.columns = columns
        self.verbose = verbose
        
        # Internal state
        self.scaler = self._initialize_scaler()
        self.fitted_columns_: List[str] = []
        self._is_fitted: bool = False
        self._logger = self._initialize_logger()
        
    def _validate_init_params(self, method: str) -> None:
        """Validate initialization parameters."""
        valid_methods = ['standard', 'minmax', 'robust']
        if method.lower() not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {method}")

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
            
        logger.info("FeatureScaler initialized with method: %s", self.method)
        return logger

    def _initialize_scaler(self):
        """Initialize the appropriate scaler object."""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler()
        elif self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

    def _validate_input_data(self, X: Any) -> pd.DataFrame:
        """Validate input data and convert to DataFrame if needed."""
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError("Input must be pandas DataFrame, Series, or numpy array")
            
        if isinstance(X, (pd.Series, np.ndarray)):
            X = pd.DataFrame(X)
            
        if X.empty:
            raise ValueError("Input data is empty")
            
        return X.copy()

    def _select_numeric_columns(self, X: pd.DataFrame) -> List[str]:
        """Select numeric columns for scaling."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            self._logger.warning("No numeric columns found in input data")
            return []
            
        if self.columns:
            # Only use specified columns that exist and are numeric
            valid_cols = [col for col in self.columns 
                         if col in numeric_cols and col in X.columns]
            if not valid_cols:
                self._logger.warning("No valid numeric columns from specified list")
            return valid_cols
            
        return numeric_cols

    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureScaler':
        """Fit the scaler to the data."""
        try:
            X = self._validate_input_data(X)
            self.fitted_columns_ = self._select_numeric_columns(X)
            
            if not self.fitted_columns_:
                self._logger.warning("No numeric columns available for scaling")
                return self
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scaler.fit(X[self.fitted_columns_])
                
            self._is_fitted = True
            self._logger.info("Successfully fitted scaler on columns: %s", self.fitted_columns_)
            
        except Exception as e:
            self._logger.error("Fitting failed: %s", str(e))
            raise
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to new data."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transformation")
            
        try:
            X = self._validate_input_data(X)
            X_transformed = X.copy()
            
            if not self.fitted_columns_:
                self._logger.warning("No columns to transform with FeatureScaler")
                return X_transformed
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_values = self.scaler.transform(X_transformed[self.fitted_columns_])
                
            # Create new columns with scaled values
            for i, col in enumerate(self.fitted_columns_):
                X_transformed[f"{col}_scaled"] = scaled_values[:, i]
                
            self._logger.debug("Successfully scaled %d features", len(self.fitted_columns_))
            return X_transformed
            
        except Exception as e:
            self._logger.error("Transformation failed: %s", str(e))
            raise

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one operation."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scaling transformation."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse transformation")
            
        try:
            X = self._validate_input_data(X)
            X_inverted = X.copy()
            
            if not self.fitted_columns_:
                return X_inverted
                
            # Extract scaled columns
            scaled_cols = [f"{col}_scaled" for col in self.fitted_columns_ 
                          if f"{col}_scaled" in X_inverted.columns]
            
            if not scaled_cols:
                self._logger.warning("No scaled columns found to inverse transform")
                return X_inverted
                
            # Get the original scaled values
            scaled_values = X_inverted[scaled_cols].values
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                original_values = self.scaler.inverse_transform(scaled_values)
                
            # Create columns with original values
            for i, col in enumerate(self.fitted_columns_):
                if f"{col}_scaled" in scaled_cols:
                    X_inverted[f"{col}_original"] = original_values[:, i]
                    
            self._logger.debug("Successfully inverse transformed %d features", len(scaled_cols))
            return X_inverted
            
        except Exception as e:
            self._logger.error("Inverse transformation failed: %s", str(e))
            raise

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Get output feature names for transformation."""
        if input_features is None:
            return [f"{col}_scaled" for col in self.fitted_columns_]
        return [f"{col}_scaled" for col in input_features if col in self.fitted_columns_]

    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'method': self.method,
            'columns': self.columns,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> 'FeatureScaler':
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        self._validate_init_params(self.method)
        return self

    def __repr__(self) -> str:
        """Official string representation of the scaler."""
        return (f"FeatureScaler(method='{self.method}', "
                f"columns={self.columns}, "
                f"verbose={self.verbose})")