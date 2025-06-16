import pandas as pd
from src.utils.logging import logger

class DataValidator:
    def __init__(self):
        logger.info("DataValidator initialized.")

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Performs basic validation on the DataFrame.
        Can be extended with more specific checks (e.g., column names, data types).
        """
        if df.empty:
            logger.warning("DataFrame is empty.")
            return False
        if df.shape[0] < 5: # Arbitrary minimum rows
            logger.warning(f"DataFrame has too few rows: {df.shape[0]}. Expected at least 5.")
            return False
        
        logger.info("Basic DataFrame validation passed.")
        return True