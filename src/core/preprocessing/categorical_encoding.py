import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logging import logger
from src.config.settings import AppSettings

class CategoricalEncoder:
    def __init__(self, ordinal_features_map: dict = None):
        self.ordinal_features_map = ordinal_features_map if ordinal_features_map is not None else AppSettings.ORDINAL_FEATURES_MAP
        self.binary_features = []
        self.nominal_features = []
        self.ordinal_features = []
        self.preprocessor = None
        self.label_encoders = {}  # To store LabelEncoders for binary features
        logger.info("CategoricalEncoder initialized.")

    def _identify_feature_types(self, X: pd.DataFrame):
        """Identifies binary, nominal, and ordinal categorical features."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.binary_features = []
        self.nominal_features = []
        self.ordinal_features = []

        for col in categorical_cols:
            unique_values = X[col].dropna().unique()
            if col in self.ordinal_features_map:
                self.ordinal_features.append(col)
            elif len(unique_values) == 2:
                self.binary_features.append(col)
            elif len(unique_values) > 2:
                self.nominal_features.append(col)
        logger.info(f"Identified Binary Features: {self.binary_features}")
        logger.info(f"Identified Nominal Features: {self.nominal_features}")
        logger.info(f"Identified Ordinal Features: {self.ordinal_features}")

    def fit(self, X: pd.DataFrame):
        """Fits the encoders based on identified feature types."""
        self._identify_feature_types(X)
        transformers = []

        # Binary Encoding (LabelEncoder)
        for col in self.binary_features:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Fitted LabelEncoder for binary feature: {col}")

        # Ordinal Encoding
        if self.ordinal_features:
            ordinal_categories = [self.ordinal_features_map[col] for col in self.ordinal_features]
            ordinal_transformer = Pipeline(steps=[
                ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('ordinal_encoder', ordinal_transformer, self.ordinal_features))
            logger.info(f"Configured OrdinalEncoder for features: {self.ordinal_features}")

        # One-Hot Encoding
        if self.nominal_features:
            onehot_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('onehot_encoder', onehot_transformer, self.nominal_features))
            logger.info(f"Configured OneHotEncoder for features: {self.nominal_features}")

        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            self.preprocessor.fit(X)
            logger.info("ColumnTransformer for categorical encoding fitted.")
        else:
            self.preprocessor = None
            logger.warning("No categorical features found or configured for encoding.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted encoders to the DataFrame."""
        X_copy = X.copy()

        # Apply Label Encoding first (manually)
        for col in self.binary_features:
            if col in X_copy.columns and col in self.label_encoders:
                X_copy[col] = self.label_encoders[col].transform(X_copy[col].astype(str))
                logger.info(f"Applied Label Encoding for {col}.")
            elif col in self.label_encoders:
                logger.warning(f"Binary feature '{col}' not found in DataFrame for transformation.")

        # Apply One-Hot and Ordinal Encoding using ColumnTransformer
        if self.preprocessor:
            cols_for_ct = [col for col in X_copy.columns if col in self.nominal_features or col in self.ordinal_features]
            if cols_for_ct:
                transformed_data = self.preprocessor.transform(X_copy)
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    feature_names_out = self.preprocessor.get_feature_names_out()
                    transformed_df = pd.DataFrame(transformed_data, index=X_copy.index, columns=feature_names_out)
                else:
                    transformed_df = pd.DataFrame(transformed_data, index=X_copy.index)
                    transformed_df.columns = [f'col_{i}' for i in range(transformed_df.shape[1])]
                passthrough_cols = [col for col in X_copy.columns if col not in self.binary_features + self.nominal_features + self.ordinal_features]
                X_copy_numeric_and_binary_encoded = X_copy.drop(columns=self.nominal_features + self.ordinal_features, errors='ignore')
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    ct_feature_names = self.preprocessor.get_feature_names_out()
                    transformed_ct_df = pd.DataFrame(transformed_data, index=X_copy.index, columns=ct_feature_names)
                    final_df = pd.concat([X_copy_numeric_and_binary_encoded[self.binary_features + passthrough_cols], transformed_ct_df], axis=1)
                else:
                    logger.error("`get_feature_names_out` not available for ColumnTransformer. Cannot reliably reconstruct DataFrame.")
                    return X_copy
            else:
                final_df = X_copy
            final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()
            logger.info("Applied categorical encoding.")
            return final_df
        else:
            logger.warning("No ColumnTransformer initialized. Returning original DataFrame.")
            return X_copy

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fits the encoder and transforms the DataFrame."""
        self.fit(X)
        return self.transform(X)