markdown
# Data Preprocessing Pipeline API Documentation

## Overview
This documentation describes the API for the data preprocessing pipeline.

## Data Loading Module
### `DataLoader`
```python
class DataLoader:
    @staticmethod
    def load_data(file_path: Union[str, Path]) -> pd.DataFrame
Loads data from various file formats into a pandas DataFrame.

Parameters:

file_path: Path to the data file (supports .csv, .txt, .xlsx, .xls, .xlsm, .json)

Returns:

Pandas DataFrame containing the loaded data

Preprocessing Modules
Missing Values
python
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame
Handles missing values by adding a flag column.

Outliers
python
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame
Detects outliers and adds a flag column.

Numeric Transformations
python
def apply_numeric_transformations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame
Applies log, standardization, and normalization to numeric columns.

Categorical Encoding
python
def encode_categorical_features(
    df: pd.DataFrame, 
    categorical_cols: List[str], 
    ordinal_mapping: Optional[Dict] = None
) -> pd.DataFrame
Encodes categorical features using appropriate encoding schemes.

Model Modules
Training
python
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    params: Optional[Dict] = None
) -> Tuple[Any, Dict]
Trains a machine learning model.

Inference
python
class InferencePipeline:
    def __init__(self, artifacts_path: Union[str, Path]):
    
    def predict(self, data: Union[pd.DataFrame, Dict]) -> Union[pd.Series, Any]
Loads saved artifacts and makes predictions on new data.

text

### 2. docs/setup.md
```markdown
# Setup Guide

## Prerequisites
- Python 3.9+
- pip
- virtualenv (recommended)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/data-preprocessing-pipeline.git
cd data-preprocessing-pipeline
2. Set up virtual environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Set up environment variables
bash
cp .env.example .env
5. Run tests
bash
make test
