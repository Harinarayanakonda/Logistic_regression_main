import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from src.utils.logging import logger

class LogTransformer:
    def __init__(self):
        self.transformer = FunctionTransformer(np.log1p, validate=False)
        self.fitted_columns = None
        logger.info("LogTransformer initialized.")

    def fit(self, X: pd.DataFrame, numeric_cols: list = None):
        """
        Identifies numeric columns to apply log transformation.
        If numeric_cols is None, automatically selects all numeric columns.
        """
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns provided for log transformation fitting.")
            self.fitted_columns = []
            return self
        
        self.fitted_columns = [col for col in numeric_cols if col in X.columns]
        logger.info(f"LogTransformer fitted on columns: {self.fitted_columns}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log1p transformation to specified numeric columns and creates new features.
        """
        X_copy = X.copy()
        if not self.fitted_columns:
            logger.warning("No columns to transform with LogTransformer. Returning original DataFrame.")
            return X_copy

        for col in self.fitted_columns:
            if col in X_copy.columns:
                new_col_name = f"{col}_log"
                # np.log1p handles zeros and negatives gracefully (returns nan for negatives)
                X_copy[new_col_name] = self.transformer.transform(X_copy[[col]])
            else:
                logger.warning(f"Column '{col}' not found for log transformation.")
        
        logger.info(f"Log transformed {len(self.fitted_columns)} features.")
        return X_copy

    def fit_transform(self, X: pd.DataFrame, numeric_cols: list = None) -> pd.DataFrame:
        self.fit(X, numeric_cols)
        return self.transform(X)