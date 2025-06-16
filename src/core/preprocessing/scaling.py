import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.logging import logger

class FeatureScaler:
    def __init__(self, method: str = 'standard'):
        """
        Initializes the FeatureScaler.
        :param method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}. Choose 'standard' or 'minmax'.")
        
        self.fitted_columns = None
        logger.info(f"FeatureScaler initialized with method: {method}")

    def fit(self, X: pd.DataFrame, columns_to_scale: list):
        """Fits the scaler on the specified numeric columns."""
        if not columns_to_scale:
            logger.warning("No columns provided for scaling fitting.")
            self.fitted_columns = []
            return self
            
        self.fitted_columns = [col for col in columns_to_scale if pd.api.types.is_numeric_dtype(X[col])]
        
        if not self.fitted_columns:
            logger.warning("No numeric columns found among provided columns_to_scale for fitting.")
            return self

        self.scaler.fit(X[self.fitted_columns])
        logger.info(f"Scaler fitted on columns: {self.fitted_columns}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies scaling to the specified numeric columns and creates new scaled features.
        """
        X_copy = X.copy()
        if not self.fitted_columns:
            logger.warning("No columns to transform with FeatureScaler. Returning original DataFrame.")
            return X_copy

        scaled_data = self.scaler.transform(X_copy[self.fitted_columns])
        for i, col_name in enumerate(self.fitted_columns):
            new_col_name = f"{col_name}_scaled"
            X_copy[new_col_name] = scaled_data[:, i]
        
        logger.info(f"Scaled {len(self.fitted_columns)} features.")
        return X_copy

    def fit_transform(self, X: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
        """
        Fits the scaler and transforms the specified columns.
        """
        self.fit(X, columns_to_scale)
        return self.transform(X)