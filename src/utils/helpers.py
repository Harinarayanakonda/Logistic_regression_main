import joblib
import pandas as pd
from datetime import datetime
from src.utils.logging import logger

def save_artifact(obj, filepath: str):
    """Saves a Python object to a file using joblib."""
    try:
        joblib.dump(obj, filepath)
        logger.info(f"Artifact saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving artifact to {filepath}: {e}")
        raise

def load_artifact(filepath: str):
    """Loads a Python object from a file using joblib."""
    try:
        obj = joblib.load(filepath)
        logger.info(f"Artifact loaded successfully from {filepath}")
        return obj
    except FileNotFoundError:
        logger.error(f"Artifact file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading artifact from {filepath}: {e}")
        raise

def generate_timestamped_filename(base_name: str, extension: str = "pkl") -> str:
    """Generates a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def get_numeric_and_categorical_features(df: pd.DataFrame):
    """Separates features into numeric and categorical lists."""
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_features, categorical_features

def display_dataframe_info(df: pd.DataFrame):
    """Displays basic information about a DataFrame."""
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info("First 5 rows:")
    logger.info(df.head())