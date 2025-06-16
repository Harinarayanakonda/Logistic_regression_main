import os

class AppSettings:
    PROJECT_NAME: str = "Logistic Regression Pipeline"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    TARGET_COLUMN: str = "target" # Placeholder, adjust based on your dataset
    VIF_THRESHOLD: float = 5.0 # Threshold for multicollinearity
    OUTLIER_IQ_FACTOR: float = 1.5 # Factor for IQR-based outlier detection

    # Ordinal encoding map example (customize for your dataset)
    # Example: {'education': {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}}
    ORDINAL_FEATURES_MAP: dict = {}

    LOGGING_LEVEL: str = "INFO"
    
    # Streamlit related settings
    STREAMLIT_APP_TITLE: str = "Logistic Regression ML Pipeline"