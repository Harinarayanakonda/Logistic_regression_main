import os
import pandas as pd
from typing import Tuple

class DataSplitter:
    def __init__(self, save_path: str):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def split_and_save(self, df: pd.DataFrame, target_column: str, index_cutoff: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        # Separate features and target
        features_df = df.drop(columns=[target_column])
        target_df = df[[target_column]]

        # Split by index
        interim_df = features_df.iloc[:index_cutoff, :]
        target_part = target_df.iloc[:index_cutoff, :]

        # Save both parts
        interim_path = os.path.join(self.save_path, "interim_preprocessed.csv")
        target_path = os.path.join(self.save_path, "target_variable.csv")

        interim_df.to_csv(interim_path, index=False)
        target_part.to_csv(target_path, index=False)

        return interim_df, target_part
