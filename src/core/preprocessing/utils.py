# src/core/preprocessing/utils.py

import pandas as pd

def separate_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Separates the target column from the DataFrame and returns feature and target DataFrames.
    """
    features_df = df.drop(columns=[target_col], errors='ignore')
    target_df = df[[target_col]] if target_col in df.columns else pd.DataFrame(index=df.index)
    return features_df, target_df

def recombine_features_and_target(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reattaches the target column to the transformed feature DataFrame using index alignment.
    """
    aligned_target_df = target_df.loc[features_df.index] if not target_df.empty else pd.DataFrame(index=features_df.index)
    return pd.concat([features_df, aligned_target_df], axis=1)
