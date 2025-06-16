import pandas as pd
import numpy as np
from src.utils.logging import logger
from src.config.settings import AppSettings

class OutlierDetector:
    def __init__(self, method: str = 'iqr', factor: float = AppSettings.OUTLIER_IQ_FACTOR):
        """
        Initializes the OutlierDetector.
        :param method: 'iqr' for Interquartile Range, 'zscore' for Z-score.
        :param factor: Factor for IQR (e.g., 1.5) or Z-score (e.g., 3).
        """
        self.method = method
        self.factor = factor
        self.lower_bounds = {}
        self.upper_bounds = {}
        logger.info(f"OutlierDetector initialized with method: {self.method}, factor: {self.factor}")

    def fit(self, X: pd.DataFrame, numeric_cols: list = None):
        """Fits the outlier detection parameters (e.g., quartiles, mean, std)."""
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns provided for outlier detection fitting.")
            return self

        for col in numeric_cols:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds[col] = Q1 - self.factor * IQR
                self.upper_bounds[col] = Q3 + self.factor * IQR
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                self.lower_bounds[col] = mean - self.factor * std
                self.upper_bounds[col] = mean + self.factor * std
            else:
                raise ValueError(f"Unsupported outlier detection method: {self.method}")
        logger.info("OutlierDetector fit completed.")
        return self

    def detect_outliers_table(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame of the same shape as X[numeric_cols], with True for outlier values.
        """
        numeric_cols = [col for col in X.select_dtypes(include=np.number).columns if col in self.lower_bounds]
        outlier_mask = pd.DataFrame(False, index=X.index, columns=numeric_cols)
        for col in numeric_cols:
            outlier_mask[col] = (X[col] < self.lower_bounds[col]) | (X[col] > self.upper_bounds[col])
        return outlier_mask

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds an 'outlier_column' flag where 1 indicates an outlier in any numeric feature, 0 otherwise.
        """
        X_copy = X.copy()
        outlier_mask = self.detect_outliers_table(X_copy)
        X_copy['outlier_column'] = outlier_mask.any(axis=1).astype(int)
        logger.info("Added 'outlier_column' flag column.")
        return X_copy

    def remove_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with rows containing any outlier removed.
        """
        outlier_mask = self.detect_outliers_table(X)
        non_outlier_rows = ~outlier_mask.any(axis=1)
        logger.info(f"Removed {(~non_outlier_rows).sum()} outlier rows.")
        return X.loc[non_outlier_rows].reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame, numeric_cols: list = None) -> pd.DataFrame:
        self.fit(X, numeric_cols)
        return self.transform(X)