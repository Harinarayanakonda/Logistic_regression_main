import pytest
import pandas as pd
import numpy as np
from src.core.preprocessing import (
    MissingValueHandler, OutlierDetector, LogTransformer, FeatureScaler,
    CategoricalEncoder, MulticollinearityReducer, PreprocessingPipelineOrchestrator
)
from src.config.settings import AppSettings

def test_missing_value_handler(sample_dataframe):
    handler = MissingValueHandler()
    df_processed = handler.fit_transform(sample_dataframe)
    assert 'Missing_Values' in df_processed.columns
    # Row 2 (index 2) and Row 5 (index 5) have NaNs in sample_dataframe
    assert df_processed.loc[2, 'Missing_Values'] == 1
    assert df_processed.loc[5, 'Missing_Values'] == 1
    # Other rows should be 0
    assert df_processed.loc[0, 'Missing_Values'] == 0
    assert df_processed.loc[1, 'Missing_Values'] == 0

def test_outlier_detector_iqr(sample_dataframe):
    detector = OutlierDetector(method='iqr')
    numeric_cols = ['numeric_col_1', 'numeric_col_2']
    detector.fit(sample_dataframe, numeric_cols)
    df_processed = detector.transform(sample_dataframe)
    assert 'outlier_column' in df_processed.columns
    # Check the known outlier (10000 in numeric_col_1 at index 6)
    assert df_processed.loc[6, 'outlier_column'] == 1
    # Check a non-outlier row
    assert df_processed.loc[0, 'outlier_column'] == 0
    # Ensure other numeric columns are handled if they had outliers
    # (assuming numeric_col_2 has no IQR outliers in sample data)
    # The flag is for ANY outlier in the row, so if only one column has it, flag is 1.

def test_log_transformer(sample_dataframe):
    transformer = LogTransformer()
    numeric_cols = ['numeric_col_1', 'numeric_col_2']
    transformer.fit(sample_dataframe, numeric_cols)
    df_processed = transformer.transform(sample_dataframe)
    assert 'numeric_col_1_log' in df_processed.columns
    assert 'numeric_col_2_log' in df_processed.columns
    # Check if values are actually transformed (log1p of 10 should be different from 10)
    assert np.isclose(df_processed.loc[0, 'numeric_col_1_log'], np.log1p(sample_dataframe.loc[0, 'numeric_col_1']))
    # NaNs should propagate if not handled, but np.log1p handles 0 gracefully.
    # The original NaN in numeric_col_1 should still be NaN in log_transformed
    assert pd.isna(df_processed.loc[2, 'numeric_col_1_log'])

def test_feature_scaler(sample_dataframe):
    # Add a log-transformed column for testing scaling on new features
    sample_df_copy = sample_dataframe.copy()
    sample_df_copy['numeric_col_1_log'] = np.log1p(sample_df_copy['numeric_col_1'])

    scaler = FeatureScaler(method='standard')
    cols_to_scale = ['numeric_col_1_log']
    scaler.fit(sample_df_copy, cols_to_scale)
    df_processed = scaler.transform(sample_df_copy)
    assert 'numeric_col_1_log_scaled' in df_processed.columns
    # Check if the mean is close to 0 and std close to 1 for the scaled column (approx)
    scaled_col = df_processed['numeric_col_1_log_scaled'].dropna()
    assert np.isclose(scaled_col.mean(), 0, atol=0.1)
    assert np.isclose(scaled_col.std(), 1, atol=0.1)

def test_categorical_encoder(sample_dataframe):
    # Define an ordinal map for testing
    AppSettings.ORDINAL_FEATURES_MAP = {'categorical_ordinal': ['Low', 'Medium', 'High']}
    encoder = CategoricalEncoder(ordinal_features_map=AppSettings.ORDINAL_FEATURES_MAP)
    
    df_processed = encoder.fit_transform(sample_dataframe)
    
    # Check Label Encoding for binary
    assert 'categorical_binary' in df_processed.columns
    # Expect 'Yes' -> 1, 'No' -> 0 (or vice versa, depends on LabelEncoder internal order)
    assert df_processed['categorical_binary'].dtype == 'int64' or df_processed['categorical_binary'].dtype == 'float64'

    # Check One-Hot Encoding for nominal
    assert 'categorical_nominal_A' in df_processed.columns
    assert 'categorical_nominal_B' in df_processed.columns
    assert 'categorical_nominal_C' in df_processed.columns
    assert 'categorical_nominal' not in df_processed.columns # Original column should be removed
    assert df_processed['categorical_nominal_A'].dtype == 'uint8' # Or bool

    # Check Ordinal Encoding for ordinal
    assert 'categorical_ordinal' in df_processed.columns # Should be transformed in place for ordinal
    assert df_processed['categorical_ordinal'].dtype == 'int64' or df_processed['categorical_ordinal'].dtype == 'float64'
    assert df_processed.loc[0, 'categorical_ordinal'] == 0 # Low
    assert df_processed.loc[1, 'categorical_ordinal'] == 1 # Medium
    assert df_processed.loc[2, 'categorical_ordinal'] == 2 # High

    # Clean up AppSettings for other tests
    AppSettings.ORDINAL_FEATURES_MAP = {}

def test_multicollinearity_reducer(sample_dataframe_for_vif):
    reducer = MulticollinearityReducer(vif_threshold=5.0)
    
    # Exclude target before fitting
    features_for_vif = [col for col in sample_dataframe_for_vif.columns if col != 'target']
    reducer.fit(sample_dataframe_for_vif[features_for_vif])
    
    df_processed = reducer.transform(sample_dataframe_for_vif)
    
    # Expect 'feature_2' to be removed due to high multicollinearity with 'feature_1'
    assert 'feature_1' in df_processed.columns
    assert 'feature_2' not in df_processed.columns
    assert 'feature_3' in df_processed.columns
    assert 'feature_4' in df_processed.columns
    assert 'target' in df_processed.columns # Non-numeric features are passed through

def test_preprocessing_pipeline_orchestrator(sample_dataframe):
    # Set target column for this test
    AppSettings.TARGET_COLUMN = 'target'
    AppSettings.ORDINAL_FEATURES_MAP = {'categorical_ordinal': ['Low', 'Medium', 'High']}

    orchestrator = PreprocessingPipelineOrchestrator()
    orchestrator.fit(sample_dataframe)
    
    processed_df = orchestrator.transform(sample_dataframe)
    
    assert 'Missing_Values' in processed_df.columns
    assert 'outlier_column' in processed_df.columns
    assert 'numeric_col_1_log' in processed_df.columns
    assert 'numeric_col_1_log_scaled' in processed_df.columns # Should have scaled log feature
    assert 'numeric_col_2_log' in processed_df.columns
    assert 'numeric_col_2_log_scaled' in processed_df.columns # Should have scaled log feature
    assert 'categorical_binary' in processed_df.columns # Label encoded
    assert 'categorical_nominal_A' in processed_df.columns # One-hot encoded
    assert 'categorical_ordinal' in processed_df.columns # Ordinal encoded
    assert 'target' in processed_df.columns # Target column should be present

    # Check if original categorical_nominal is removed
    assert 'categorical_nominal' not in processed_df.columns
    
    # Check if the number of features is reasonable after all transformations
    # This might depend on the specific data and VIF results, but it shouldn't be empty.
    assert processed_df.shape[1] > 5
    
    # Check if get_feature_names_after_preprocessing works
    final_features = orchestrator.get_feature_names_after_preprocessing()
    assert isinstance(final_features, list)
    assert len(final_features) == (processed_df.shape[1] -1) # Excluding target if it's there
    assert all(f in processed_df.columns for f in final_features)
    
    # Clean up AppSettings for other tests
    AppSettings.TARGET_COLUMN = 'target' # Reset to default if changed
    AppSettings.ORDINAL_FEATURES_MAP = {}