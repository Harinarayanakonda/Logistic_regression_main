import pandas as pd
from src.utils.helpers import load_artifact
from src.utils.logging import logger
from src.config.paths import PREPROCESSING_ARTIFACTS_DIR, TRAINED_MODELS_DIR
import os

class ModelInferrer:
    def __init__(self):
        self.preprocessing_pipeline = None
        self.model = None
        logger.info("ModelInferrer initialized.")

    def load_inference_artifacts(self, model_artifact_path: str):
        """Loads the saved preprocessing pipeline and trained model."""
        try:
            full_artifact = load_artifact(model_artifact_path)
            self.preprocessing_pipeline = full_artifact['preprocessing_pipeline']
            self.model = full_artifact['model']
            logger.info(f"Preprocessing pipeline and model loaded from {model_artifact_path}")
        except Exception as e:
            logger.error(f"Failed to load inference artifacts from {model_artifact_path}: {e}")
            raise

    def predict(self, raw_data: pd.DataFrame) -> pd.Series:
        """
        Performs inference on new raw data using the loaded preprocessing pipeline and model.
        """
        if self.preprocessing_pipeline is None or self.model is None:
            raise RuntimeError("Preprocessing pipeline or model not loaded. Call load_inference_artifacts first.")

        logger.info(f"Starting inference on {raw_data.shape[0]} samples.")
        
        # Apply preprocessing pipeline
        # The preprocessing pipeline in `pipeline_orchestrator` directly processes the dataframe.
        # It's not an sklearn.pipeline.Pipeline object
        try:
            # First, check if input columns match what the preprocessing pipeline expects (best effort)
            # This requires the preprocessing pipeline to store expected feature names.
            # For now, we trust the pipeline to handle unseen columns or reorder.
            
            processed_data = self.preprocessing_pipeline.transform(raw_data.copy())
            logger.info(f"Data processed for inference. Shape: {processed_data.shape}")

            # Ensure the processed_data has columns matching the model's expected input
            # This is crucial. The multicollinearity reducer stores `selected_features`.
            # We must ensure `processed_data` only contains these features, in the correct order.
            
            expected_features = self.preprocessing_pipeline.get_feature_names_after_preprocessing()
            
            # Filter and reorder processed_data to match expected features
            if set(expected_features).issubset(set(processed_data.columns)):
                processed_data_aligned = processed_data[expected_features]
                logger.info("Processed data aligned with model's expected features.")
            else:
                missing_features = set(expected_features) - set(processed_data.columns)
                if missing_features:
                    logger.error(f"Missing features in processed data for inference: {missing_features}")
                    raise ValueError(f"Processed data lacks expected features for inference: {missing_features}")
                # If some features are extra, filter them out.
                processed_data_aligned = processed_data[expected_features]


            predictions = self.model.predict(processed_data_aligned)
            logger.info("Inference completed.")
            return pd.Series(predictions, name="prediction", index=raw_data.index)
        
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def predict_proba(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs probability prediction on new raw data using the loaded preprocessing pipeline and model.
        """
        if self.preprocessing_pipeline is None or self.model is None:
            raise RuntimeError("Preprocessing pipeline or model not loaded. Call load_inference_artifacts first.")
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model does not support predict_proba. Returning None.")
            return None

        logger.info(f"Starting probability inference on {raw_data.shape[0]} samples.")
        
        try:
            processed_data = self.preprocessing_pipeline.transform(raw_data.copy())
            
            expected_features = self.preprocessing_pipeline.get_feature_names_after_preprocessing()
            if set(expected_features).issubset(set(processed_data.columns)):
                processed_data_aligned = processed_data[expected_features]
            else:
                missing_features = set(expected_features) - set(processed_data.columns)
                if missing_features:
                    logger.error(f"Missing features in processed data for predict_proba: {missing_features}")
                    raise ValueError(f"Processed data lacks expected features for predict_proba: {missing_features}")
                processed_data_aligned = processed_data[expected_features]

            probabilities = self.model.predict_proba(processed_data_aligned)
            
            # Assuming binary classification for class 0 and class 1
            if len(self.model.classes_) == 2:
                proba_df = pd.DataFrame(probabilities, columns=['probability_0', 'probability_1'], index=raw_data.index)
            else:
                proba_df = pd.DataFrame(probabilities, columns=[f'probability_{c}' for c in self.model.classes_], index=raw_data.index)
            
            logger.info("Probability inference completed.")
            return proba_df
        
        except Exception as e:
            logger.error(f"Error during probability inference: {e}")
            raise