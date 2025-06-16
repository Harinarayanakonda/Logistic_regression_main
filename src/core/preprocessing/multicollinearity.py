import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.utils.logging import logger
from src.config.settings import AppSettings

class MulticollinearityReducer:
    def __init__(self, vif_threshold: float = AppSettings.VIF_THRESHOLD):
        """
        Initializes the MulticollinearityReducer with a specified VIF threshold.
        """
        self.vif_threshold = vif_threshold
        self.selected_features = None
        logger.info(f"MulticollinearityReducer initialized with VIF threshold: {self.vif_threshold}")

    def fit(self, X: pd.DataFrame):
        """
        Identifies features to keep based on VIF.
        Assumes X contains only numerical features or features already encoded.

        Parameters:
            X (pd.DataFrame): The input DataFrame to fit on.

        Returns:
            self
        """
        X_numerical = X.select_dtypes(include=['number'])
        if X_numerical.empty:
            logger.warning("No numerical features to check for multicollinearity.")
            self.selected_features = X.columns.tolist()
            return self

        features = X_numerical.columns.tolist()
        df_for_vif = X_numerical.dropna().copy()

        if df_for_vif.empty:
            logger.warning("DataFrame is empty after dropping NaNs, cannot calculate VIF. Keeping all original features.")
            self.selected_features = X.columns.tolist()
            return self

        while True:
            vif_data = pd.DataFrame()
            vif_data["feature"] = features
            try:
                vif_data["VIF"] = [variance_inflation_factor(df_for_vif[features].values, i)
                                   for i in range(df_for_vif[features].shape[1])]
            except np.linalg.LinAlgError as e:
                logger.warning(f"Singular matrix encountered during VIF calculation: {e}. Cannot calculate VIF, keeping current features.")
                break
            except Exception as e:
                logger.warning(f"An error occurred during VIF calculation: {e}. Keeping current features.")
                break

            max_vif = vif_data["VIF"].max()
            if max_vif > self.vif_threshold:
                feature_to_remove = vif_data.loc[vif_data['VIF'] == max_vif, 'feature'].iloc[0]
                features.remove(feature_to_remove)
                logger.info(f"Removed '{feature_to_remove}' due to high VIF: {max_vif:.2f}")
                if not features:
                    logger.warning("All features removed due to multicollinearity. This might indicate an issue with the data or threshold.")
                    break
            else:
                break

        original_non_numeric_features = X.select_dtypes(exclude=['number']).columns.tolist()
        self.selected_features = features + original_non_numeric_features
        logger.info(f"Selected features after multicollinearity reduction: {self.selected_features}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with only the features selected after multicollinearity reduction.

        The selected features are determined during the fit step using VIF analysis.
        If the reducer has not been fitted, returns a copy of the original DataFrame.

        Parameters:
            X (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame containing only the features selected to avoid multicollinearity.
        """
        if self.selected_features is None:
            logger.warning("MulticollinearityReducer not fitted. Returning original DataFrame.")
            return X.copy()

        existing_selected_features = [col for col in self.selected_features if col in X.columns]
        if not existing_selected_features:
            logger.warning("No selected features found in the current DataFrame after transformation.")
            return pd.DataFrame(index=X.index)
        return X[existing_selected_features].copy()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the reducer and returns the DataFrame with reduced multicollinearity.

        Parameters:
            X (pd.DataFrame): The input DataFrame to fit and transform.

        Returns:
            pd.DataFrame: DataFrame containing only the features selected to avoid multicollinearity.
        """
        self.fit(X)
        return self.transform(X)