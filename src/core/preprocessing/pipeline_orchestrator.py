import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
# Explicitly import sklearn components used in mocks for clarity,
# though their direct use is within the mock classes
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.feature_selection import VarianceThreshold # Not directly used in VIF mock
from scipy.stats import zscore # Used in OutlierDetector (mock)
from statsmodels.stats.outliers_influence import variance_inflation_factor # Used in MulticollinearityReducerWrapper (real VIF)

# --- Mock AppSettings and Logger ---

# Mock AppSettings for demonstration purposes
class AppSettings:
    TARGET_COLUMN = "target" # Default target column name
    OUTLIER_IQ_FACTOR = 1.5  # IQR factor for outlier detection
    OUTLIER_ZSCORE_FACTOR = 3.0 # Z-score factor for outlier detection
    VIF_THRESHOLD = 5.0 # VIF threshold for multicollinearity reduction
    # Example: Map for ordinal features. Keys are column names, values are lists of ordered categories.
    ORDINAL_FEATURES_MAP = {
        'categorical_col_C': ['A', 'B', 'C'], # Example for mock data
    }

# Mock logger for demonstration purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Using mock implementations for AppSettings, logger, and various preprocessing/feature engineering classes for demonstration.")


# --- Mock paths and Helper Functions ---
PROCESSED_DATA_DIR = "processed_data"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def generate_timestamped_filename(prefix: str, suffix: str = ".csv") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{suffix}"

def download_dataframe(df: pd.DataFrame, filename: str, mime_type: str = "text/csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime=mime_type,
        key=f"download_{filename.replace('.', '_')}"
    )

# --- Core Preprocessing Steps (Base Classes and Individual Transformers) ---

class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base class for all custom transformers to enforce common methods and logging."""
    def __init__(self, name="BaseTransformer"):
        self.name = name

    def fit(self, X, y=None):
        logger.info(f"Fitting {self.name}...")
        return self

    def transform(self, X):
        logger.info(f"Transforming with {self.name}...")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input to {self.name} must be a pandas DataFrame.")
        if X.empty:
            logger.warning(f"Input DataFrame is empty for {self.name}. Returning empty DataFrame.")
            return pd.DataFrame(columns=X.columns) # Return empty DataFrame with original columns
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# --- Individual Preprocessing Step Wrappers (using scikit-learn's BaseEstimator) ---
# These are the classes the Orchestrator will directly use.
# They wrap the specific logic (MissingValueHandler, OutlierDetector etc.)

class MissingValuesTransformer(BaseTransformer):
    def __init__(self):
        super().__init__("MissingValuesTransformer")
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.numeric_cols = []
        self.categorical_cols = []
        self.high_missing_cols_to_drop = []

    def fit(self, X, y=None):
        super().fit(X, y)
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.high_missing_cols_to_drop = []

        # Identify columns with too many NaNs to drop entirely
        for col in X.columns:
            if X[col].isnull().mean() > 0.3 or X[col].isnull().all(): # Over 30% missing or all NaNs
                self.high_missing_cols_to_drop.append(col)
                logger.info(f"Column '{col}' identified for dropping due to high missing percentage.")

        # Filter columns for imputation (don't fit on cols that will be dropped)
        cols_for_imputation_numeric = [col for col in self.numeric_cols if col not in self.high_missing_cols_to_drop]
        cols_for_imputation_categorical = [col for col in self.categorical_cols if col not in self.high_missing_cols_to_drop]

        if cols_for_imputation_numeric:
            self.numeric_imputer.fit(X[cols_for_imputation_numeric])
        if cols_for_imputation_categorical:
            self.categorical_imputer.fit(X[cols_for_imputation_categorical])
            
        logger.info(f"MissingValuesTransformer fitted. Numeric: {len(cols_for_imputation_numeric)}, Categorical: {len(cols_for_imputation_categorical)}. Dropping: {len(self.high_missing_cols_to_drop)}.")
        return self

    def transform(self, X):
        super().transform(X)
        X_transformed = X.copy()

        # Drop identified high-missing columns first
        cols_to_drop_present = [col for col in self.high_missing_cols_to_drop if col in X_transformed.columns]
        if cols_to_drop_present:
            X_transformed = X_transformed.drop(columns=cols_to_drop_present)
            logger.info(f"Dropped high-missing columns during transform: {cols_to_drop_present}")

        # Impute numeric columns
        cols_to_impute_numeric_present = [col for col in self.numeric_cols if col in X_transformed.columns]
        if cols_to_impute_numeric_present and hasattr(self, 'numeric_imputer') and self.numeric_imputer.statistics_ is not None:
            X_transformed[cols_to_impute_numeric_present] = self.numeric_imputer.transform(X_transformed[cols_to_impute_numeric_present])
            logger.info(f"Imputed numeric columns: {cols_to_impute_numeric_present}")

        # Impute categorical columns
        cols_to_impute_categorical_present = [col for col in self.categorical_cols if col in X_transformed.columns]
        if cols_to_impute_categorical_present and hasattr(self, 'categorical_imputer') and self.categorical_imputer.statistics_ is not None:
            X_transformed[cols_to_impute_categorical_present] = self.categorical_imputer.transform(X_transformed[cols_to_impute_categorical_present])
            logger.info(f"Imputed categorical columns: {cols_to_impute_categorical_present}")

        # Add missing flags for columns that originally had NaNs and were imputed
        for col in X.columns:
            if col not in cols_to_drop_present and X[col].isnull().any():
                flag_col_name = f"{col}_Missing_Flag"
                if flag_col_name not in X_transformed.columns: # Avoid adding flag if already exists
                    X_transformed[flag_col_name] = X[col].isnull().astype(int)
                    logger.info(f"Added missing flag for column: {col}")

        logger.info(f"MissingValuesTransformer transformed. New shape: {X_transformed.shape}")
        return X_transformed


class OutlierDetectorTransformer(BaseTransformer):
    def __init__(self, method='iqr', factor=1.5):
        super().__init__("OutlierDetectorTransformer")
        if method not in ['iqr', 'zscore']:
            raise ValueError(f"Unsupported outlier detection method: {method}. Choose 'iqr' or 'zscore'.")
        self.method = method
        self.factor = factor
        self.bounds = {} # Stores min/max or IQR bounds for each feature

    def fit(self, X, y=None):
        super().fit(X, y)
        self.bounds = {}
        numeric_cols = X.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if X[col].isnull().all():
                logger.warning(f"Column '{col}' contains all NaNs. Skipping outlier detection fitting for this column.")
                continue
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                self.bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                self.bounds[col] = {'mean': mean, 'std': std}
        logger.info(f"OutlierDetectorTransformer fitted for {len(self.bounds)} numeric columns.")
        return self

    def transform(self, X):
        super().transform(X)
        X_transformed = X.copy()
        for col, bounds in self.bounds.items():
            if col in X_transformed.columns and pd.api.types.is_numeric_dtype(X_transformed[col]):
                outlier_flag_col = f"{col}_Outlier_Flag"
                # Ensure the flag column is not duplicated if it already exists from a previous run on some data
                if outlier_flag_col in X_transformed.columns:
                    logger.debug(f"Outlier flag column '{outlier_flag_col}' already exists. Overwriting.")

                if self.method == 'iqr':
                    X_transformed[outlier_flag_col] = ((X_transformed[col] < bounds['lower']) |
                                                        (X_transformed[col] > bounds['upper'])).astype(int)
                elif self.method == 'zscore':
                    mean = bounds['mean']
                    std = bounds['std']
                    # Handle case where standard deviation is zero (all values are same)
                    if std == 0:
                        X_transformed[outlier_flag_col] = 0 # No outliers if no variance
                    else:
                        X_transformed[outlier_flag_col] = (np.abs((X_transformed[col] - mean) / std) > self.factor).astype(int)
                logger.info(f"Added outlier flag for column: {col} using {self.method} method.")
            elif col not in X_transformed.columns:
                logger.warning(f"Column '{col}' not found in transform data for OutlierDetectorTransformer. Skipping outlier detection.")
        logger.info(f"OutlierDetectorTransformer transformed. Shape: {X_transformed.shape}")
        return X_transformed

class LogTransformerWrapper(BaseTransformer):
    def __init__(self, threshold_for_skewness_std=1.0, exclude_cols=None):
        super().__init__("LogTransformerWrapper")
        # A simple heuristic: if std dev is very low relative to mean, maybe not worth transforming
        # This is a very basic heuristic; typically skewness (scipy.stats.skew) is used
        self.threshold_for_skewness_std = threshold_for_skewness_std
        self.transformed_cols = []
        self.exclude_cols = exclude_cols if exclude_cols is not None else []

    def fit(self, X, y=None):
        super().fit(X, y)
        self.transformed_cols = []
        numeric_cols = X.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col in self.exclude_cols:
                continue
            
            # Skip if column has any negative values, as log1p expects x >= -1 (x >= 0 for typical data)
            if X[col].min() < 0:
                logger.warning(f"Column '{col}' contains negative values. Skipping log transformation for this column.")
                continue

            # Heuristic for skewness: if mean is notably different from median AND std is above a threshold
            # A more robust check would involve scipy.stats.skew
            if X[col].mean() > X[col].median() and X[col].std() > self.threshold_for_skewness_std:
                self.transformed_cols.append(col)
                logger.info(f"Column '{col}' selected for log transformation (heuristic: mean > median and std > {self.threshold_for_skewness_std}).")
            else:
                logger.debug(f"Column '{col}' not selected for log transformation (heuristic failed).")
        logger.info(f"LogTransformerWrapper fitted. Will transform {len(self.transformed_cols)} columns.")
        return self

    def transform(self, X):
        super().transform(X)
        X_transformed = X.copy()
        renamed_cols = {}
        for col in self.transformed_cols:
            if col in X_transformed.columns:
                # Apply log1p transformation: log(1 + x). This handles x=0 gracefully.
                X_transformed[col] = np.log1p(X_transformed[col])
                new_col_name = f"{col}_log"
                # Check if the new column name already exists to prevent overwrite issues
                if new_col_name not in X_transformed.columns:
                    X_transformed = X_transformed.rename(columns={col: new_col_name})
                    renamed_cols[col] = new_col_name
                else:
                    logger.warning(f"Renamed column '{new_col_name}' already exists. Overwriting original '{col}' directly.")
                    # If new_col_name exists, we simply update the values in place
                    # but don't rename the column to avoid KeyError later if old name is used.
                    # Or, a more robust strategy might be to create a truly unique name.
                    # For now, let's assume renaming is safe or the new column name is unique.
                    # If we don't rename, the original column name would still be present.
                    # A better way is to create a new column, then drop the old one.
                    X_transformed[new_col_name] = X_transformed[col]
                    X_transformed = X_transformed.drop(columns=[col])
                    renamed_cols[col] = new_col_name
            else:
                logger.warning(f"Column '{col}' not found in transform data for LogTransformerWrapper. Skipping transformation.")

        logger.info(f"LogTransformerWrapper transformed. Shape: {X_transformed.shape}")
        return X_transformed

class FeatureScalerWrapper(BaseTransformer):
    def __init__(self, method='standard'):
        super().__init__("FeatureScalerWrapper")
        self.method = method
        self.scaler = None
        self.columns_to_scale = []

    def fit(self, X, y=None):
        super().fit(X, y)
        # Select only numeric columns that have variance (std > 0)
        numeric_cols = X.select_dtypes(include=np.number).columns
        self.columns_to_scale = [col for col in numeric_cols if X[col].std() > 1e-9] # Avoid scaling constant columns

        if not self.columns_to_scale:
            logger.info("No numeric columns with variance to scale. Skipping FeatureScalerWrapper fitting.")
            self.scaler = None
            return self

        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.method}")

        self.scaler.fit(X[self.columns_to_scale])
        logger.info(f"FeatureScalerWrapper fitted for {len(self.columns_to_scale)} numeric columns.")
        return self

    def transform(self, X):
        super().transform(X)
        if self.scaler is None or not self.columns_to_scale:
            logger.info("No scaler fitted or no columns to scale. Returning original DataFrame.")
            return X.copy()

        X_transformed = X.copy()
        # Ensure only columns that were fitted are transformed and are present in current DF
        cols_present_to_scale = [col for col in self.columns_to_scale if col in X_transformed.columns]
        
        if cols_present_to_scale:
            # Create a temporary DataFrame for scaling to preserve original dtypes if needed
            scaled_data = self.scaler.transform(X_transformed[cols_present_to_scale])
            X_transformed[cols_present_to_scale] = scaled_data
            logger.info(f"Scaled {len(cols_present_to_scale)} columns using {self.method} method.")
        else:
            logger.warning("No fitted numeric columns found in transform data for FeatureScalerWrapper. Returning DataFrame without scaling numeric cols.")

        logger.info(f"FeatureScalerWrapper transformed. Shape: {X_transformed.shape}")
        return X_transformed


class CategoricalEncoderWrapper(BaseTransformer):
    def __init__(self, ordinal_features_map=None, handle_unknown='ignore'):
        super().__init__("CategoricalEncoderWrapper")
        self.ordinal_features_map = ordinal_features_map if ordinal_features_map is not None else {}
        self.one_hot_encoder = None
        self.ordinal_encoder = None
        self.ohe_features = [] # Features to be One-Hot Encoded
        self.ord_features = [] # Features to be Ordinal Encoded
        self.handle_unknown = handle_unknown # 'ignore' is good for production to avoid errors on unseen categories

    def fit(self, X, y=None):
        super().fit(X, y)
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        self.ohe_features = []
        self.ord_features = []

        for col in categorical_cols:
            if col in self.ordinal_features_map:
                self.ord_features.append(col)
            else:
                self.ohe_features.append(col)

        if self.ohe_features:
            # Convert categorical columns to 'category' dtype if not already, to ensure consistent categories
            X_ohe_fit = X[self.ohe_features].astype('category')
            self.one_hot_encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
            self.one_hot_encoder.fit(X_ohe_fit)
            logger.info(f"OneHotEncoder fitted for {len(self.ohe_features)} columns. Categories: {[c.categories_ for c in self.one_hot_encoder.categories_]}")
        else:
            self.one_hot_encoder = None

        if self.ord_features:
            # Prepare categories for OrdinalEncoder
            categories_for_ordinal = [self.ordinal_features_map[col] for col in self.ord_features]
            # Convert ordinal columns to 'category' dtype before fitting
            X_ord_fit = X[self.ord_features].astype('category')
            self.ordinal_encoder = OrdinalEncoder(categories=categories_for_ordinal, handle_unknown=self.handle_unknown)
            self.ordinal_encoder.fit(X_ord_fit)
            logger.info(f"OrdinalEncoder fitted for {len(self.ord_features)} columns.")
        else:
            self.ordinal_encoder = None

        return self

    def transform(self, X):
        super().transform(X)
        X_transformed = X.copy()
        
        # Handle One-Hot Encoding
        if self.one_hot_encoder and self.ohe_features:
            cols_to_ohe_present = [col for col in self.ohe_features if col in X_transformed.columns]
            if cols_to_ohe_present:
                # Convert to category dtype for consistency before transform
                X_ohe_transform = X_transformed[cols_to_ohe_present].astype('category')
                ohe_output = self.one_hot_encoder.transform(X_ohe_transform)
                ohe_feature_names = self.one_hot_encoder.get_feature_names_out(cols_to_ohe_present)
                ohe_df = pd.DataFrame(ohe_output, index=X_transformed.index, columns=ohe_feature_names)
                X_transformed = pd.concat([X_transformed.drop(columns=cols_to_ohe_present), ohe_df], axis=1)
                logger.info(f"Applied One-Hot Encoding to {len(cols_to_ohe_present)} columns. New shape: {X_transformed.shape}")
            else:
                logger.warning("No One-Hot Encoded columns found in transform data for CategoricalEncoderWrapper. Skipping OHE.")

        # Handle Ordinal Encoding
        if self.ordinal_encoder and self.ord_features:
            cols_to_ord_present = [col for col in self.ord_features if col in X_transformed.columns]
            if cols_to_ord_present:
                # Convert to category dtype for consistency before transform
                X_ord_transform = X_transformed[cols_to_ord_present].astype('category')
                X_transformed[cols_to_ord_present] = self.ordinal_encoder.transform(X_ord_transform)
                logger.info(f"Applied Ordinal Encoding to {len(cols_to_ord_present)} columns.")
            else:
                logger.warning("No Ordinal Encoded columns found in transform data for CategoricalEncoderWrapper. Skipping Ordinal Encoding.")
                
        logger.info(f"CategoricalEncoderWrapper transformed. Shape: {X_transformed.shape}")
        return X_transformed


class MulticollinearityReducerWrapper(BaseTransformer):
    def __init__(self, threshold=5.0):
        super().__init__("MulticollinearityReducerWrapper")
        self.threshold = threshold
        self.columns_to_drop = []
        self._fitted_columns_order = None # Store order of columns seen during fit

    def fit(self, X, y=None):
        super().fit(X, y)
        self.columns_to_drop = []
        numeric_df = X.select_dtypes(include=np.number)
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            logger.info("Not enough numeric columns for VIF calculation. Skipping MulticollinearityReducerWrapper fitting.")
            self._fitted_columns_order = []
            return self

        # Drop columns with zero variance before VIF calculation to avoid division by zero
        variances = numeric_df.var()
        zero_variance_cols = variances[variances == 0].index.tolist()
        if zero_variance_cols:
            logger.warning(f"Columns with zero variance detected and will be excluded from VIF: {zero_variance_cols}")
            numeric_df = numeric_df.drop(columns=zero_variance_cols)

        if numeric_df.empty or numeric_df.shape[1] < 2: # Re-check after dropping zero-variance
            logger.info("Not enough numeric columns remaining after removing zero variance columns. Skipping VIF.")
            self._fitted_columns_order = []
            return self

        self._fitted_columns_order = numeric_df.columns.tolist() # Store order for consistent VIF calculation

        # Calculate VIF iteratively
        columns_for_vif = list(numeric_df.columns)
        while True:
            if len(columns_for_vif) < 2: # Need at least two columns to calculate VIF
                break

            try:
                # Ensure the DataFrame used for VIF calculation only contains `columns_for_vif`
                current_numeric_subset = numeric_df[columns_for_vif]
                
                vif_data = pd.DataFrame()
                vif_data["feature"] = current_numeric_subset.columns
                vif_data["VIF"] = [
                    variance_inflation_factor(current_numeric_subset.values, i)
                    for i in range(current_numeric_subset.shape[1])
                ]
            except np.linalg.LinAlgError as e:
                logger.warning(f"Singular matrix encountered during VIF calculation: {e}. Cannot calculate VIF for remaining columns.")
                break # Exit loop if matrix is singular
            except ValueError as e: # Handle cases where VIF computation might fail for other reasons
                logger.warning(f"ValueError during VIF calculation: {e}. This often happens with perfect multicollinearity. Stopping VIF reduction.")
                break

            max_vif_row = vif_data.loc[vif_data['VIF'].idxmax()]
            max_vif = max_vif_row['VIF']
            col_to_remove = max_vif_row['feature']

            if max_vif > self.threshold:
                self.columns_to_drop.append(col_to_remove)
                columns_for_vif.remove(col_to_remove)
                logger.info(f"Dropped '{col_to_remove}' due to VIF = {max_vif:.2f} > {self.threshold}")
            else:
                break # All remaining VIFs are below threshold

        logger.info(f"MulticollinearityReducerWrapper fitted. Will drop {len(self.columns_to_drop)} columns.")
        return self

    def transform(self, X):
        super().transform(X)
        X_transformed = X.copy()
        
        # Only drop columns that were identified during fit and are present in the current DF
        cols_to_drop_present = [col for col in self.columns_to_drop if col in X_transformed.columns]

        if cols_to_drop_present:
            X_transformed = X_transformed.drop(columns=cols_to_drop_present)
            logger.info(f"MulticollinearityReducerWrapper transformed. Dropped {len(cols_to_drop_present)} columns. Shape: {X_transformed.shape}")
        else:
            logger.info("No multicollinear columns to drop in transform data for MulticollinearityReducerWrapper. Returning original DataFrame.")

        return X_transformed


    # --- Main PreprocessingPipelineOrchestrator ---
class PreprocessingPipelineOrchestrator:
    def __init__(self, target_column: str = AppSettings.TARGET_COLUMN,
                 ordinal_features_map: dict = AppSettings.ORDINAL_FEATURES_MAP):
        self.target_column = target_column
        self.ordinal_features_map = ordinal_features_map
        self.pipeline_steps = {}  # Store fitted transformers
        self.fitted_raw_feature_names = None
        self._target_column_present_during_fit = False
        self._final_feature_columns_after_fit = None
        
        # Initialize session state tracking if not exists
        if 'preprocessing_steps_completed' not in st.session_state:
            st.session_state.preprocessing_steps_completed = {
                'missing_values': False,
                'outliers': False,
                'log_transform': False,
                'scaling': False,
                'encoding': False,
                'multicollinearity': False
            }
        
        # Internal tracking for pipeline steps
        self._completed_steps = set()
        self._intermediate_data = {}
        
        logger.info("PreprocessingPipelineOrchestrator initialized with Streamlit session state integration.")

    def _check_dataframe_not_empty(self, df: pd.DataFrame, step_name: str):
        """Helper to check if DataFrame is empty and log a warning/error."""
        if df.empty:
            logger.error(f"DataFrame became empty after '{step_name}' step. "
                        f"Current shape: {df.shape}. This will cause issues for subsequent steps.")
            raise ValueError(f"DataFrame became empty after '{step_name}' step. Cannot proceed.")
        return True

    def is_step_completed(self, step_name: str) -> bool:
        """Check if a specific preprocessing step has been completed."""
        # Check both internal and session state completion
        step_mapping = {
            'missing_handler': 'missing_values',
            'outlier_detector': 'outliers',
            'log_transformer': 'log_transform',
            'scaler': 'scaling',
            'encoder': 'encoding',
            'multicollinearity_reducer': 'multicollinearity'
        }
        
        if step_name in step_mapping:
            return (step_name in self._completed_steps and 
                    st.session_state.preprocessing_steps_completed[step_mapping[step_name]])
        return step_name in self._completed_steps

    def get_completed_steps(self) -> list:
        """Return list of completed step names."""
        return list(self._completed_steps)

    def mark_step_completed(self, step_name: str):
        """Mark a step as completed in both orchestrator and session state."""
        self._completed_steps.add(step_name)
        
        # Map internal step names to UI step names
        step_mapping = {
            'missing_handler': 'missing_values',
            'outlier_detector': 'outliers',
            'log_transformer': 'log_transform',
            'scaler': 'scaling',
            'encoder': 'encoding',
            'multicollinearity_reducer': 'multicollinearity'
        }
        
        if step_name in step_mapping:
            st.session_state.preprocessing_steps_completed[step_mapping[step_name]] = True
        
        logger.info(f"Marked step '{step_name}' as completed in both orchestrator and session state")

    def get_intermediate_data(self, step_name: str) -> pd.DataFrame:
        """Get intermediate data after a specific step."""
        return self._intermediate_data.get(step_name, None)

    def _run_single_step(self, step_name: str, input_df: pd.DataFrame) -> pd.DataFrame:
        """Execute a single preprocessing step."""
        step_methods = {
            'missing_handler': (self.pipeline_steps['missing_handler'], "Missing Values"),
            'outlier_detector': (self.pipeline_steps['outlier_detector'], "Outlier Detection"),
            'log_transformer': (self.pipeline_steps['log_transformer'], "Log Transformation"),
            'scaler': (self.pipeline_steps['scaler'], "Scaling"),
            'encoder': (self.pipeline_steps['encoder'], "Categorical Encoding"),
            'multicollinearity_reducer': (self.pipeline_steps['multicollinearity_reducer'], "Multicollinearity Reduction")
        }

        if step_name not in step_methods:
            raise ValueError(f"Unknown step name: {step_name}")

        transformer, step_label = step_methods[step_name]
        output_df = transformer.transform(input_df)
        self._check_dataframe_not_empty(output_df, step_label)
        self._intermediate_data[step_name] = output_df.copy()
        self.mark_step_completed(step_name)
        logger.info(f"Completed single step '{step_name}'. Shape: {output_df.shape}")
        return output_df

    def fit(self, df: pd.DataFrame):
        """Fits the entire preprocessing pipeline."""
        df_copy = df.copy()
        self.fitted_raw_feature_names = df_copy.columns.tolist()
        self._completed_steps.clear()
        self._intermediate_data.clear()

        # Reset session state completion flags
        for key in st.session_state.preprocessing_steps_completed:
            st.session_state.preprocessing_steps_completed[key] = False

        # Separate target temporarily if it's present
        features_df = df_copy.drop(columns=[self.target_column], errors='ignore')
        if self.target_column in df_copy.columns:
            self._target_column_present_during_fit = True
        else:
            self._target_column_present_during_fit = False
            logger.warning(f"Target column '{self.target_column}' not found during fit.")

        logger.info(f"Initial features_df shape for fit: {features_df.shape}")

        # Step 1: Missing Value Handling
        missing_handler = MissingValuesTransformer()
        features_df = missing_handler.fit_transform(features_df)
        self._check_dataframe_not_empty(features_df, "Missing Values")
        self._intermediate_data['missing_handler'] = features_df.copy()
        self.mark_step_completed('missing_handler')
        logger.info(f"Shape after Missing Values (fit): {features_df.shape}")

        # Step 2: Outlier Detection
        outlier_detector = OutlierDetectorTransformer(method='iqr', factor=AppSettings.OUTLIER_IQ_FACTOR)
        features_df = outlier_detector.fit_transform(features_df)
        self._check_dataframe_not_empty(features_df, "Outlier Detection")
        self._intermediate_data['outlier_detector'] = features_df.copy()
        self.mark_step_completed('outlier_detector')
        logger.info(f"Shape after Outlier Detection (fit): {features_df.shape}")

        # Step 3: Log Transformation
        log_transformer = LogTransformerWrapper()
        features_df = log_transformer.fit_transform(features_df)
        self._check_dataframe_not_empty(features_df, "Log Transformation")
        self._intermediate_data['log_transformer'] = features_df.copy()
        self.mark_step_completed('log_transformer')
        logger.info(f"Shape after Log Transformation (fit): {features_df.shape}")

        # Step 4: Scaling
        scaler = FeatureScalerWrapper(method='standard')
        features_df = scaler.fit_transform(features_df)
        self._check_dataframe_not_empty(features_df, "Scaling")
        self._intermediate_data['scaler'] = features_df.copy()
        self.mark_step_completed('scaler')
        logger.info(f"Shape after Scaling (fit): {features_df.shape}")

        # Step 5: Categorical Encoding
        encoder = CategoricalEncoderWrapper(ordinal_features_map=self.ordinal_features_map)
        features_df = encoder.fit_transform(features_df)
        self._check_dataframe_not_empty(features_df, "Categorical Encoding")
        self._intermediate_data['encoder'] = features_df.copy()
        self.mark_step_completed('encoder')
        logger.info(f"Shape after Categorical Encoding (fit): {features_df.shape}")

        # Step 6: Multicollinearity Reduction
        reducer = MulticollinearityReducerWrapper(threshold=AppSettings.VIF_THRESHOLD)
        features_df = reducer.fit_transform(features_df)
        self._check_dataframe_not_empty(features_df, "Multicollinearity Reduction")
        self._intermediate_data['multicollinearity_reducer'] = features_df.copy()
        self.mark_step_completed('multicollinearity_reducer')
        logger.info(f"Shape after Multicollinearity Reduction (fit): {features_df.shape}")

        # Store fitted transformers
        self.pipeline_steps = {
            'missing_handler': missing_handler,
            'outlier_detector': outlier_detector,
            'log_transformer': log_transformer,
            'scaler': scaler,
            'encoder': encoder,
            'multicollinearity_reducer': reducer
        }

        # Store final feature columns
        self._final_feature_columns_after_fit = features_df.columns.tolist()

        logger.info("Preprocessing pipeline fitted successfully.")
        return self

    # ... (keep all other existing methods unchanged)

    def transform(self, df: pd.DataFrame, steps_to_run: list = None) -> pd.DataFrame:
        """
        Transforms the DataFrame using the fitted pipeline.
        
        Args:
            df: Input DataFrame
            steps_to_run: List of steps to run (None runs all steps)
                         Possible values: ['missing_handler', 'outlier_detector', 
                         'log_transformer', 'scaler', 'encoder', 'multicollinearity_reducer']
        """
        if not self.pipeline_steps:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")

        default_steps_order = [
            'missing_handler',
            'outlier_detector',
            'log_transformer',
            'scaler',
            'encoder',
            'multicollinearity_reducer'
        ]

        steps_to_run = steps_to_run or default_steps_order
        df_transformed = df.copy()

        # Separate target if present
        features_df_input = df_transformed.drop(columns=[self.target_column], errors='ignore')
        target_series_input = None
        if self.target_column in df_transformed.columns:
            target_series_input = df_transformed[self.target_column]

        # Align features to what pipeline was fitted on
        aligned_features_df = self._align_input_features(features_df_input)
        logger.info(f"Aligned features shape: {aligned_features_df.shape}")

        # Run requested steps in order
        processed_features_df = aligned_features_df.copy()
        for step in steps_to_run:
            if step not in default_steps_order:
                raise ValueError(f"Invalid step name: {step}")
            processed_features_df = self._run_single_step(step, processed_features_df)

        # Final alignment and target recombination
        df_final_output = self._finalize_output(processed_features_df, target_series_input)
        return df_final_output

    def _align_input_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Align input features to what pipeline was fitted on."""
        aligned_df = pd.DataFrame(index=input_df.index)
        expected_raw_features = [col for col in self.fitted_raw_feature_names 
                               if col != self.target_column]

        for col in expected_raw_features:
            if col in input_df.columns:
                aligned_df[col] = input_df[col]
            else:
                aligned_df[col] = np.nan

        return aligned_df.reindex(columns=expected_raw_features, fill_value=np.nan)

    def _finalize_output(self, processed_df: pd.DataFrame, 
                        target_series: pd.Series = None) -> pd.DataFrame:
        """Ensure final output has correct columns and optional target."""
        df_final = pd.DataFrame(index=processed_df.index)

        for col in self._final_feature_columns_after_fit:
            if col in processed_df.columns:
                df_final[col] = processed_df[col]
            else:
                df_final[col] = 0.0

        df_final = df_final.reindex(columns=self._final_feature_columns_after_fit, fill_value=0.0)

        if target_series is not None:
            df_final[self.target_column] = target_series.loc[df_final.index]
            cols = [col for col in df_final.columns if col != self.target_column] + [self.target_column]
            df_final = df_final[cols]

        return df_final

    def get_feature_names_after_preprocessing(self) -> list:
        """Returns the list of feature names after full preprocessing, excluding the target column."""
        if self._final_feature_columns_after_fit is None:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")
        return [col for col in self._final_feature_columns_after_fit if col != self.target_column]

    def get_raw_feature_names(self) -> list:
        """Returns the list of raw feature names the pipeline was fitted on."""
        if self.fitted_raw_feature_names is None:
            raise RuntimeError("Pipeline not fitted. Call .fit() first to determine raw feature names.")
        return self.fitted_raw_feature_names

    def save(self, path: str):
        """Saves the fitted PreprocessingPipelineOrchestrator to a file."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Preprocessing pipeline saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving preprocessing pipeline to {path}: {e}")
            raise

    @classmethod
    def load(cls, path: str):
        """Loads a fitted PreprocessingPipelineOrchestrator from a file."""
        try:
            with open(path, 'rb') as f:
                orchestrator_instance = pickle.load(f)
            if not isinstance(orchestrator_instance, cls):
                raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
            logger.info(f"Preprocessing pipeline loaded successfully from {path}")
            return orchestrator_instance
        except Exception as e:
            logger.error(f"Error loading preprocessing pipeline from {path}: {e}")
            raise