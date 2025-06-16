import os
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
INTERIM_DATA_DIR = DATA_DIR / "02_interim"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"
MODEL_INPUTS_DIR = DATA_DIR / "04_model_inputs"

# Model and artifact directories
MODELS_DIR = ROOT_DIR / "models"
PREPROCESSING_ARTIFACTS_DIR = MODELS_DIR / "preprocessing_artifacts"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"

# Ensure directories exist
for path_dir in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_INPUTS_DIR,
    PREPROCESSING_ARTIFACTS_DIR,
    TRAINED_MODELS_DIR,
]:
    path_dir.mkdir(parents=True, exist_ok=True)