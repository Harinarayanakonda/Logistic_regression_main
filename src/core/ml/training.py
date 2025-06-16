import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.settings import AppSettings
from src.config.paths import MODEL_INPUTS_DIR
from src.utils.logging import logger
import os

class ModelTrainer:
    def __init__(self, target_column: str = AppSettings.TARGET_COLUMN, random_state: int = AppSettings.RANDOM_STATE, test_size: float = AppSettings.TEST_SIZE):
        self.target_column = target_column
        self.random_state = random_state
        self.test_size = test_size
        logger.info("ModelTrainer initialized.")

    def split_and_save_data(self, df: pd.DataFrame):
        """
        Splits the DataFrame into training and testing sets and saves them to CSV.
        """
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found in DataFrame.")
            raise ValueError(f"Target column '{self.target_column}' missing.")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y # Stratify if classification
        )

        # Save datasets
        X_train_path = os.path.join(MODEL_INPUTS_DIR, "X_train.csv")
        X_test_path = os.path.join(MODEL_INPUTS_DIR, "X_test.csv")
        y_train_path = os.path.join(MODEL_INPUTS_DIR, "y_train.csv")
        y_test_path = os.path.join(MODEL_INPUTS_DIR, "y_test.csv")

        X_train.to_csv(X_train_path, index=False, header=True)
        X_test.to_csv(X_test_path, index=False, header=True)
        y_train.to_csv(y_train_path, index=False, header=True)
        y_test.to_csv(y_test_path, index=False, header=True)

        logger.info("Data split into training and testing sets and saved to CSV files in 'data/04_model_inputs'.")
        logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, model):
        """
        Trains the given model. This is a placeholder for your actual model training logic.
        """
        logger.info(f"Starting model training for {model.__class__.__name__}...")
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        return model