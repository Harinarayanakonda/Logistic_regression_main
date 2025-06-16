import pandas as pd
from sklearn.impute import MissingIndicator
from src.utils.logging import logger

class MissingValueHandler:
    def __init__(self):
        self.indicator = None
        logger.info("MissingValueHandler initialized.")

    def fit(self, X: pd.DataFrame):
        """Fits the MissingIndicator on the DataFrame."""
        logger.info("MissingValueHandler fit completed.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a 'Missing_Values' column where 1 indicates any missing value in the row, 0 otherwise.
        """
        X_copy = X.copy()
        missing_row_flag = X_copy.isnull().any(axis=1)
        X_copy['Missing_Values'] = missing_row_flag.astype(int)
        logger.info("Added 'Missing_Values' flag column.")
        return X_copy

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms the DataFrame."""
        self.fit(X)
        return self.transform(X)