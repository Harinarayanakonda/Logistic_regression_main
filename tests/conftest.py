import pytest
import pandas as pd
import numpy as np
import os
from src.config.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODEL_INPUTS_DIR, PREPROCESSING_ARTIFACTS_DIR, TRAINED_MODELS_DIR

@pytest.fixture(scope="session", autouse=True)
def setup_test_directories():
    """Ensure test directories are clean and exist before tests run."""
    for path_dir in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_INPUTS_DIR,
        PREPROCESSING_ARTIFACTS_DIR,
        TRAINED_MODELS_DIR,
    ]:
        if os.path.exists(path_dir):
            for item in os.listdir(path_dir):
                item_path = os.path.join(path_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
        path_dir.mkdir(parents=True, exist_ok=True)
    yield
    # Teardown: You might choose to clean up directories after tests, but often
    # it's better to inspect test outputs if tests fail.
    # For CI, you might want to remove them.

@pytest.fixture(scope="module")
def sample_dataframe():
    """A sample DataFrame for testing preprocessing steps."""
    data = {
        'numeric_col_1': [10, 20, np.nan, 40, 50, 100, 1000, 2, 3, 15],
        'numeric_col_2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
        'categorical_binary': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'categorical_nominal': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'categorical_ordinal': ['Low', 'Medium', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium'],
        'has_missing': [1, 0, 1, 0, 0, 0, 0, 0, 0, 0], # Placeholder for missing flag
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    # Introduce some NaN for testing missing value handling
    df.loc[2, 'numeric_col_1'] = np.nan
    df.loc[5, 'categorical_nominal'] = np.nan
    # Introduce an outlier for testing outlier detection
    df.loc[6, 'numeric_col_1'] = 10000 
    return df

@pytest.fixture(scope="module")
def sample_dataframe_for_vif():
    """A sample DataFrame with multicollinearity for VIF testing."""
    data = {
        'feature_1': [10, 12, 15, 18, 20],
        'feature_2': [20, 24, 30, 36, 40], # highly correlated with feature_1 (2 * feature_1)
        'feature_3': [5, 6, 7, 8, 9],
        'feature_4': [100, 105, 110, 115, 120], # some other feature
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture(scope="module")
def mock_artifact_path(tmp_path_factory):
    """Provides a temporary path for saving/loading artifacts."""
    return tmp_path_factory.mktemp("artifacts")