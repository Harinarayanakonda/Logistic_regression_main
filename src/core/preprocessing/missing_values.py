import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from typing import List, Optional, Union, Dict
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MissingValueHandler:
    """
    Enhanced missing value handler with multiple imputation strategies.
    Ensures only one missing flag is created per feature.
    """

    def __init__(self,
                 numeric_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 add_missing_flag: bool = True,
                 knn_neighbors: int = 5,
                 drop_high_missing_threshold: float = 0.4):
        if not 0 <= drop_high_missing_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.add_missing_flag = add_missing_flag
        self.knn_neighbors = knn_neighbors
        self.drop_high_missing_threshold = drop_high_missing_threshold

        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.imputation_values_ = {}
        self.imputed_counts_ = {}
        self.dropped_columns_ = []
        self._imputation_summary = pd.DataFrame(columns=["Column", "Imputation Method", "Details"])
        self.imputation_log = []

        logger.info("MissingValueHandler initialized")

    def _get_high_missing_cols_to_drop(self, X: pd.DataFrame) -> List[str]:
        missing_ratio = X.isnull().mean()
        high_missing_cols = missing_ratio[missing_ratio >= self.drop_high_missing_threshold].index.tolist()
        all_nan_cols = X.columns[X.isnull().all()].tolist()
        for col in all_nan_cols:
            if col not in high_missing_cols:
                high_missing_cols.append(col)
        return high_missing_cols

    def _identify_columns(self, X: pd.DataFrame):
        self.numeric_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.categorical_cols_ = [col for col in self.categorical_cols_ if col not in self.numeric_cols_]

    def _store_imputation_parameters(self, col_type: str, cols_used: List[str], imputer):
        if hasattr(imputer, 'statistics_'):
            for i, col in enumerate(cols_used):
                self.imputation_values_[col] = imputer.statistics_[i]
        elif col_type == 'numeric' and self.numeric_strategy == 'knn':
            self.imputation_values_[f"KNN_Numeric_Strategy"] = {'n_neighbors': self.knn_neighbors}
        else:
            logger.warning(f"Could not store imputation values for {col_type}")

    def fit(self, X: pd.DataFrame) -> 'MissingValueHandler':
        self.imputation_log.append("Starting missing value handling process")
        self.dropped_columns_ = self._get_high_missing_cols_to_drop(X)
        if self.dropped_columns_:
            self.imputation_log.append(f"Dropping columns with >{self.drop_high_missing_threshold*100}% missing: {self.dropped_columns_}")

        X_for_fit = X.drop(columns=self.dropped_columns_, errors='ignore')
        self._identify_columns(X_for_fit)

        if self.numeric_cols_:
            cols_with_nans = [col for col in self.numeric_cols_ if X_for_fit[col].isnull().any()]
            if cols_with_nans:
                if self.numeric_strategy == 'knn':
                    self.numeric_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
                    self.imputation_log.append(f"Using KNN imputer (k={self.knn_neighbors}) for numeric columns")
                else:
                    self.numeric_imputer_ = SimpleImputer(strategy=self.numeric_strategy)
                    self.imputation_log.append(f"Using {self.numeric_strategy} imputation for numeric columns")
                self.numeric_imputer_.fit(X_for_fit[cols_with_nans])
                self._store_imputation_parameters('numeric', cols_with_nans, self.numeric_imputer_)

        if self.categorical_cols_:
            cols_with_nans = [col for col in self.categorical_cols_ if X_for_fit[col].isnull().any()]
            if cols_with_nans:
                self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_strategy)
                self.imputation_log.append(f"Using {self.categorical_strategy} imputation for categorical columns")
                self.categorical_imputer_.fit(X_for_fit[cols_with_nans])
                self._store_imputation_parameters('categorical', cols_with_nans, self.categorical_imputer_)

        self._imputation_summary = self._generate_imputation_summary_df()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        self.imputed_counts_ = {}
        self.imputation_log.append("Starting transformation")

        relevant_cols = [col for col in X_copy.columns if col not in self.dropped_columns_]
        original_missing_flags = X_copy[relevant_cols].isna()

        if self.dropped_columns_:
            X_copy = X_copy.drop(columns=self.dropped_columns_)
            self.imputation_log.append(f"Dropped columns: {self.dropped_columns_}")

        numeric_cols_to_impute = [col for col in self.numeric_cols_ if col in X_copy.columns and X_copy[col].isnull().any()]
        if numeric_cols_to_impute and self.numeric_imputer_:
            before_counts = X_copy[numeric_cols_to_impute].isnull().sum()
            X_copy[numeric_cols_to_impute] = self.numeric_imputer_.transform(X_copy[numeric_cols_to_impute])
            after_counts = X_copy[numeric_cols_to_impute].isnull().sum()
            for col in numeric_cols_to_impute:
                self.imputed_counts_[col] = before_counts[col] - after_counts[col]
            self.imputation_log.append(f"Imputed {len(numeric_cols_to_impute)} numeric columns")

        categorical_cols_to_impute = [col for col in self.categorical_cols_ if col in X_copy.columns and X_copy[col].isnull().any()]
        if categorical_cols_to_impute and self.categorical_imputer_:
            before_counts = X_copy[categorical_cols_to_impute].isnull().sum()
            X_copy[categorical_cols_to_impute] = self.categorical_imputer_.transform(X_copy[categorical_cols_to_impute])
            after_counts = X_copy[categorical_cols_to_impute].isnull().sum()
            for col in categorical_cols_to_impute:
                self.imputed_counts_[col] = before_counts[col] - after_counts[col]
            self.imputation_log.append(f"Imputed {len(categorical_cols_to_impute)} categorical columns")

        # Add missing flags (ensure only one flag per original feature)
        if self.add_missing_flag:
            for col in original_missing_flags.columns:
                if original_missing_flags[col].any():
                    flag_col = f"{col}_Missing_Flag"
                    X_copy[flag_col] = original_missing_flags[col].astype(int)
            self.imputation_log.append("Added missing value flags")

        remaining_nas = X_copy.isna().sum().sum()
        if remaining_nas > 0:
            self.imputation_log.append(f"Warning: {remaining_nas} missing values remain after imputation")

        return X_copy

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # Remaining methods unchanged (summary, log, save/load)...

    @property
    def imputation_summary(self) -> pd.DataFrame:
        """Get summary DataFrame matching UI requirements."""
        if self._imputation_summary.empty:
            self._imputation_summary = self._generate_imputation_summary_df()
        return self._imputation_summary

    def get_imputation_log(self) -> List[str]:
        """Get detailed imputation log for UI display."""
        return self.imputation_log

    def _generate_imputation_summary_df(self) -> pd.DataFrame:
        """Generate summary DataFrame with columns: [Column, Imputation Method, Details]."""
        records = []
        
        # Dropped columns
        for col in self.dropped_columns_:
            records.append({
                "Column": col,
                "Imputation Method": "Dropped",
                "Details": f"Over {self.drop_high_missing_threshold*100:.0f}% missing"
            })
        
        # Imputed columns
        for col, value in self.imputation_values_.items():
            if isinstance(value, dict) and 'n_neighbors' in value:  # KNN case
                records.append({
                    "Column": col,
                    "Imputation Method": "KNN Imputation",
                    "Details": f"n_neighbors={value['n_neighbors']}"
                })
            else:
                method = (f"{self.numeric_strategy.capitalize()} Imputation" 
                          if col in self.numeric_cols_ 
                          else f"{self.categorical_strategy.capitalize()} Imputation")
                
                details = (f"Imputed with {value:.2f}" if isinstance(value, (int, float)) 
                          else f"Imputed with '{value}'")
                
                if col in self.imputed_counts_:
                    details += f" ({self.imputed_counts_[col]} values imputed)"
                
                records.append({
                    "Column": col,
                    "Imputation Method": method,
                    "Details": details
                })
        
        # Missing flags
        if self.add_missing_flag and (self.dropped_columns_ or self.imputation_values_):
            records.append({
                "Column": "Missing Flags",
                "Imputation Method": "Flag Added",
                "Details": "Binary flags for original missingness"
            })
        
        return pd.DataFrame(records)

    def save(self, filepath: str):
        """Save handler state to file."""
        state = {
            'numeric_imputer_': self.numeric_imputer_,
            'categorical_imputer_': self.categorical_imputer_,
            'numeric_cols_': self.numeric_cols_,
            'categorical_cols_': self.categorical_cols_,
            'imputation_values_': self.imputation_values_,
            'dropped_columns_': self.dropped_columns_,
            '_imputation_summary': self._imputation_summary,
            'imputation_log': self.imputation_log,
            'params': {
                'numeric_strategy': self.numeric_strategy,
                'categorical_strategy': self.categorical_strategy,
                'add_missing_flag': self.add_missing_flag,
                'knn_neighbors': self.knn_neighbors,
                'drop_high_missing_threshold': self.drop_high_missing_threshold
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved handler state to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MissingValueHandler':
        """Load handler from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        handler = cls(**state['params'])
        
        # Restore state
        handler.numeric_imputer_ = state.get('numeric_imputer_', None)
        handler.categorical_imputer_ = state.get('categorical_imputer_', None)
        handler.numeric_cols_ = state.get('numeric_cols_', [])
        handler.categorical_cols_ = state.get('categorical_cols_', [])
        handler.imputation_values_ = state.get('imputation_values_', {})
        handler.dropped_columns_ = state.get('dropped_columns_', [])
        handler._imputation_summary = state.get('_imputation_summary', pd.DataFrame())
        handler.imputation_log = state.get('imputation_log', [])

        return handler