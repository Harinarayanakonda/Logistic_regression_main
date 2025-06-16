import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logging import logger
from src.utils.helpers import get_numeric_and_categorical_features
from .missing_values import MissingValueHandler
from .outlier_detection import OutlierDetector
from .log_transform import LogTransformer
from .scaling import FeatureScaler
from .categorical_encoding import CategoricalEncoder
from .multicollinearity import MulticollinearityReducer
from src.config.settings import AppSettings

# Custom Transformer for MissingValueHandler to fit into Pipeline
class MissingValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.handler = MissingValueHandler()

    def fit(self, X, y=None):
        return self.handler.fit(X)

    def transform(self, X):
        return self.handler.transform(X)

# Custom Transformer for OutlierDetector
class OutlierDetectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', factor=AppSettings.OUTLIER_IQ_FACTOR):
        self.detector = OutlierDetector(method=method, factor=factor)
        self.numeric_cols = None

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        return self.detector.fit(X, self.numeric_cols)

    def transform(self, X):
        return self.detector.transform(X)

# Custom Transformer for LogTransformer
class LogTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.log_transformer = LogTransformer()
        self.numeric_cols = None

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        return self.log_transformer.fit(X, self.numeric_cols)

    def transform(self, X):
        return self.log_transformer.transform(X)

# Custom Transformer for FeatureScaler
class FeatureScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, method='standard'):
        self.scaler = FeatureScaler(method=method)
        self.cols_to_scale = None # These will be log-transformed features

    def fit(self, X, y=None):
        # We assume log transformation has already happened, so select the new log-transformed columns
        log_cols = [col for col in X.columns if col.endswith('_log')]
        original_numeric_cols = [col for col in X.select_dtypes(include=['number']).columns.tolist() if not col.endswith('_log')]
        
        # Decide which columns to scale. For this pipeline, we scale newly created log features
        # If no log features, scale original numeric features that are not target
        self.cols_to_scale = log_cols if log_cols else original_numeric_cols
        
        # Exclude the target column from scaling
        if AppSettings.TARGET_COLUMN in self.cols_to_scale:
            self.cols_to_scale.remove(AppSettings.TARGET_COLUMN)

        return self.scaler.fit(X, self.cols_to_scale)

    def transform(self, X):
        return self.scaler.transform(X)

# Custom Transformer for MulticollinearityReducer
class MulticollinearityReducerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, vif_threshold: float = AppSettings.VIF_THRESHOLD):
        self.reducer = MulticollinearityReducer(vif_threshold=vif_threshold)

    def fit(self, X, y=None):
        # The multicollinearity reducer needs all *numerical* features that are candidates for modeling.
        # This will be applied *after* all transformations and encodings.
        return self.reducer.fit(X)

    def transform(self, X):
        return self.reducer.transform(X)

class PreprocessingPipelineOrchestrator:
    def __init__(self, target_column: str = AppSettings.TARGET_COLUMN, 
                 ordinal_features_map: dict = AppSettings.ORDINAL_FEATURES_MAP):
        self.target_column = target_column
        self.ordinal_features_map = ordinal_features_map
        self.pipeline = None
        self.original_columns_at_multicollinearity_step = None # To store columns before VIF for debugging
        logger.info("PreprocessingPipelineOrchestrator initialized.")

    def fit(self, df: pd.DataFrame):
        """Fits the entire preprocessing pipeline."""
        df_copy = df.copy()

        # Step 1: Missing Value Handling
        missing_handler = MissingValuesTransformer()
        df_copy = missing_handler.fit_transform(df_copy)
        logger.info(f"Shape after Missing Values: {df_copy.shape}")

        # Step 2: Outlier Detection
        outlier_detector = OutlierDetectorTransformer()
        df_copy = outlier_detector.fit_transform(df_copy)
        logger.info(f"Shape after Outlier Detection: {df_copy.shape}")

        # Step 3: Log Transformation
        log_transformer = LogTransformerWrapper()
        df_copy = log_transformer.fit_transform(df_copy)
        logger.info(f"Shape after Log Transformation: {df_copy.shape}")

        # Step 4: Scaling (on log-transformed features or original numeric)
        scaler = FeatureScalerWrapper()
        df_copy = scaler.fit_transform(df_copy)
        logger.info(f"Shape after Scaling: {df_copy.shape}")
        
        # Store non-target numeric columns which will be inputs to the Multicollinearity Reducer
        self.original_columns_at_multicollinearity_step = [
            col for col in df_copy.select_dtypes(include=['number']).columns if col != self.target_column
        ]

        # Step 5: Categorical Encoding
        encoder = CategoricalEncoder(ordinal_features_map=self.ordinal_features_map)
        df_copy = encoder.fit_transform(df_copy)
        logger.info(f"Shape after Categorical Encoding: {df_copy.shape}")
        
        # Step 6: Multicollinearity Reduction
        # This step needs to operate on the fully transformed and encoded DataFrame.
        # It's a selection process, not a transformation in the sklearn Pipeline sense.
        # So we'll run it explicitly here after all other steps.
        reducer = MulticollinearityReducer()
        # Filter out the target column before fitting reducer
        features_for_vif = [col for col in df_copy.columns if col != self.target_column]
        reducer.fit(df_copy[features_for_vif])
        self.pipeline_steps = {
            'missing_handler': missing_handler,
            'outlier_detector': outlier_detector,
            'log_transformer': log_transformer,
            'scaler': scaler,
            'encoder': encoder,
            'multicollinearity_reducer': reducer # Store fitted reducer
        }
        logger.info("Preprocessing pipeline fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the DataFrame using the fitted pipeline."""
        if not hasattr(self, 'pipeline_steps') or self.pipeline_steps is None:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")

        df_transformed = df.copy()

        df_transformed = self.pipeline_steps['missing_handler'].transform(df_transformed)
        df_transformed = self.pipeline_steps['outlier_detector'].transform(df_transformed)
        df_transformed = self.pipeline_steps['log_transformer'].transform(df_transformed)
        df_transformed = self.pipeline_steps['scaler'].transform(df_transformed)
        df_transformed = self.pipeline_steps['encoder'].transform(df_transformed)
        
        # Apply multicollinearity reduction as the last step
        features_for_vif = [col for col in df_transformed.columns if col != self.target_column]
        df_transformed = self.pipeline_steps['multicollinearity_reducer'].transform(df_transformed)
        
        logger.info(f"Shape after full preprocessing: {df_transformed.shape}")
        return df_transformed

    def get_feature_names_after_preprocessing(self) -> list:
        """Returns the list of feature names after full preprocessing."""
        if not hasattr(self, 'pipeline_steps') or self.pipeline_steps is None:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")
        
        # This is tricky because the last step (multicollinearity) is a selection
        # and doesn't explicitly modify the sklearn pipeline output.
        # The `reducer.selected_features` holds the final columns.
        return self.pipeline_steps['multicollinearity_reducer'].selected_features