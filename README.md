# Data Preprocessing Pipeline

A professional, production-grade data preprocessing pipeline in Python.

## Features
- Support for multiple data formats (CSV, Excel, JSON, etc.)
- Comprehensive data cleaning and preprocessing
- Automated feature engineering
- Model training and inference pipelines
- Interactive Streamlit web interface

## Installation
See [setup guide](docs/setup.md) for detailed installation instructions.

## Usage
```python
from data.raw_data.data_loader import DataLoader
from preprocessing.pipeline import PreprocessingPipeline

# Load data
df = DataLoader.load_data("data/raw/data.csv")

# Preprocess data
pipeline = PreprocessingPipeline()
processed_df = pipeline.run(df)