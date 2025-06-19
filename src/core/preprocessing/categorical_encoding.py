import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    KBinsDiscretizer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional, Union, Any
import logging
import pickle
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans  # For potential high-cardinality clustering
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Production-grade categorical variable encoder with comprehensive features:
    
    Key Features:
    - Automatic detection of binary/nominal/ordinal/high-cardinality features
    - Manual strategy override via column_encoding_strategy
    - Robust handling of unseen categories with configurable strategies
    - Multiple high-cardinality strategies (frequency, clustering, target-encoding)
    - Comprehensive input validation and error handling
    - Full scikit-learn estimator API compliance
    - Detailed logging and documentation
    - Memory-efficient transformations
    - Serialization/deserialization support
    - Feature names tracking
    
    Parameters:
        column_encoding_strategy: Optional dict of {col: strategy} to override auto-detection
        ordinal_features_map: Dict of {col: ordered_categories} for ordinal encoding
        max_cardinality: Threshold for high-cardinality treatment (default: 20)
        high_cardinality_strategy: Strategy for high-cardinality features 
                                  ('frequency', 'clustering', or 'target')
        handle_unknown: How to handle unseen categories ('ignore', 'error', or 'impute')
        unknown_value: Value to use for unknown categories when handle_unknown='impute'
        target_col: Optional target column name for target encoding strategy
        random_state: Random seed for reproducible clustering
    """

    VALID_HIGH_CARDINALITY_STRATEGIES = ['frequency', 'clustering', 'target', 'none']
    VALID_HANDLE_UNKNOWN = ['ignore', 'error', 'impute']

    def __init__(self,
                 column_encoding_strategy: Optional[Dict[str, str]] = None,
                 ordinal_features_map: Optional[Dict[str, List]] = None,
                 max_cardinality: int = 20,
                 high_cardinality_strategy: str = 'frequency',
                 handle_unknown: str = 'ignore',
                 unknown_value: Any = -1,
                 target_col: Optional[str] = None,
                 random_state: Optional[int] = None):
        
        self.column_encoding_strategy = column_encoding_strategy or {}
        self.ordinal_features_map = ordinal_features_map or {}
        self.max_cardinality = max_cardinality
        self.high_cardinality_strategy = high_cardinality_strategy
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.target_col = target_col
        self.random_state = random_state

        # Validate parameters
        self._validate_init_params()

        # Initialize feature tracking
        self.binary_features: List[str] = []
        self.nominal_features: List[str] = []
        self.ordinal_features: List[str] = []
        self.high_cardinality_features: List[str] = []
        self.numerical_features: List[str] = []

        # Initialize encoders
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.bin_encoders: Dict[str, Union[KBinsDiscretizer, KMeans]] = {}
        self.target_encoders: Dict[str, Dict[str, float]] = {}
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None

        logger.info("CategoricalEncoder initialized with parameters: "
                   f"max_cardinality={max_cardinality}, "
                   f"high_cardinality_strategy={high_cardinality_strategy}, "
                   f"handle_unknown={handle_unknown}")

    def _validate_init_params(self):
        """Validate initialization parameters."""
        if not isinstance(self.max_cardinality, int) or self.max_cardinality <= 0:
            raise ValueError("max_cardinality must be a positive integer")
            
        if self.high_cardinality_strategy not in self.VALID_HIGH_CARDINALITY_STRATEGIES:
            raise ValueError(
                f"Invalid high_cardinality_strategy. Must be one of: {self.VALID_HIGH_CARDINALITY_STRATEGIES}"
            )
            
        if self.handle_unknown not in self.VALID_HANDLE_UNKNOWN:
            raise ValueError(
                f"Invalid handle_unknown. Must be one of: {self.VALID_HANDLE_UNKNOWN}"
            )
            
        if self.high_cardinality_strategy == 'target' and not self.target_col:
            raise ValueError("target_col must be specified when using high_cardinality_strategy='target'")

    def _identify_feature_types(self, X: pd.DataFrame) -> None:
        """Identify feature types based on data characteristics and user overrides."""
        # Clear previous identifications
        self.binary_features.clear()
        self.nominal_features.clear()
        self.ordinal_features.clear()
        self.high_cardinality_features.clear()
        self.numerical_features.clear()

        # Get all columns excluding target if specified
        all_cols = X.columns.tolist()
        if self.target_col and self.target_col in all_cols:
            all_cols.remove(self.target_col)

        # Separate numerical and categorical columns
        numerical_cols = X[all_cols].select_dtypes(include=['number']).columns.tolist()
        categorical_cols = list(set(all_cols) - set(numerical_cols))

        self.numerical_features = numerical_cols

        for col in categorical_cols:
            # Skip target column if specified
            if col == self.target_col:
                continue

            n_unique = len(X[col].dropna().unique())
            explicit_encoding = self.column_encoding_strategy.get(col)

            if explicit_encoding:
                self._apply_explicit_encoding(col, explicit_encoding, n_unique)
            else:
                self._auto_detect_encoding(col, n_unique)

        logger.info(
            f"Feature types identified - Binary: {self.binary_features}, "
            f"Ordinal: {self.ordinal_features}, Nominal: {self.nominal_features}, "
            f"High-Cardinality: {self.high_cardinality_features}, "
            f"Numerical: {self.numerical_features}"
        )

    def _apply_explicit_encoding(self, col: str, encoding: str, n_unique: int) -> None:
        """Apply user-specified encoding strategy for a column."""
        if encoding.lower() == 'binary':
            if n_unique > 2:
                warnings.warn(
                    f"Column '{col}' has {n_unique} unique values but was marked as binary. "
                    "Only the first two values will be encoded."
                )
            self.binary_features.append(col)
        elif encoding.lower() == 'ordinal':
            if col not in self.ordinal_features_map:
                warnings.warn(
                    f"Column '{col}' marked as ordinal but no mapping provided. "
                    "Will use default ordinal encoding."
                )
            self.ordinal_features.append(col)
        elif encoding.lower() == 'onehot':
            self.nominal_features.append(col)
        elif encoding.lower() == 'high_cardinality':
            self.high_cardinality_features.append(col)
        else:
            raise ValueError(f"Unknown encoding strategy '{encoding}' for column '{col}'")

    def _auto_detect_encoding(self, col: str, n_unique: int) -> None:
        """Automatically detect encoding strategy for a column."""
        if col in self.ordinal_features_map:
            self.ordinal_features.append(col)
        elif n_unique == 2:
            self.binary_features.append(col)
        elif n_unique > self.max_cardinality:
            self.high_cardinality_features.append(col)
        else:
            self.nominal_features.append(col)

    def _fit_binary_encoding(self, X: pd.DataFrame) -> None:
        """Fit LabelEncoders for binary features."""
        for col in self.binary_features:
            le = LabelEncoder()
            try:
                # Handle potential mixed types by converting to string
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Fitted LabelEncoder for binary feature: {col}")
            except Exception as e:
                raise ValueError(f"Error fitting LabelEncoder for binary feature '{col}': {str(e)}")

    def _fit_high_cardinality_encoding(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit encoders for high-cardinality features based on strategy."""
        if not self.high_cardinality_features:
            return

        if self.high_cardinality_strategy == 'frequency':
            self._fit_frequency_encoding(X)
        elif self.high_cardinality_strategy == 'clustering':
            self._fit_cluster_encoding(X)
        elif self.high_cardinality_strategy == 'target' and y is not None:
            self._fit_target_encoding(X, y)

    def _fit_frequency_encoding(self, X: pd.DataFrame) -> None:
        """Fit frequency-based encoding for high-cardinality features."""
        for col in self.high_cardinality_features:
            n_unique = len(X[col].dropna().unique())
            n_bins = min(10, n_unique)
            
            try:
                binner = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode='ordinal',
                    strategy='quantile'
                )
                # KBins expects 2D input
                binner.fit(X[[col]])
                self.bin_encoders[col] = binner
                logger.info(
                    f"Fitted KBinsDiscretizer for high-cardinality feature: {col} "
                    f"with {binner.n_bins_[0]} bins"
                )
            except Exception as e:
                raise ValueError(
                    f"Error fitting KBinsDiscretizer for high-cardinality feature '{col}': {str(e)}"
                )

    def _fit_cluster_encoding(self, X: pd.DataFrame) -> None:
        """Fit cluster-based encoding for high-cardinality features."""
        # Note: This is a placeholder implementation
        # In production, you would want to properly one-hot encode before clustering
        # and handle the dimensionality appropriately
        warnings.warn(
            "Cluster encoding for high-cardinality features is experimental. "
            "Consider using frequency or target encoding instead."
        )
        
        for col in self.high_cardinality_features:
            try:
                # Simple implementation - would need enhancement for production
                # Convert categories to numerical representation first
                le = LabelEncoder()
                encoded = le.fit_transform(X[col].astype(str)).reshape(-1, 1)
                
                # Determine optimal number of clusters (simple heuristic)
                n_unique = len(le.classes_)
                n_clusters = min(10, max(2, int(np.sqrt(n_unique))))
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state
                )
                kmeans.fit(encoded)
                self.bin_encoders[col] = (le, kmeans)
                logger.info(
                    f"Fitted KMeans clustering for high-cardinality feature: {col} "
                    f"with {n_clusters} clusters"
                )
            except Exception as e:
                raise ValueError(
                    f"Error fitting cluster encoder for high-cardinality feature '{col}': {str(e)}"
                )

    def _fit_target_encoding(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit target encoding for high-cardinality features."""
        if y is None:
            raise ValueError("Target required for target encoding but none provided")
            
        for col in self.high_cardinality_features:
            try:
                # Calculate mean target per category
                target_map = X.groupby(col)[y.name].mean().to_dict()
                self.target_encoders[col] = target_map
                logger.info(
                    f"Fitted target encoding for high-cardinality feature: {col} "
                    f"with {len(target_map)} categories"
                )
            except Exception as e:
                raise ValueError(
                    f"Error fitting target encoder for high-cardinality feature '{col}': {str(e)}"
                )

    def _fit_nominal_ordinal_encoding(self, X: pd.DataFrame) -> None:
        """Fit ColumnTransformer for nominal and ordinal features."""
        transformers = []

        # Ordinal encoding
        if self.ordinal_features:
            ordinal_categories = []
            for col in self.ordinal_features:
                if col in self.ordinal_features_map:
                    # Use user-provided categories
                    ordinal_categories.append(self.ordinal_features_map[col])
                else:
                    # Infer categories from data
                    ordinal_categories.append(list(X[col].dropna().unique()))
            
            ordinal_pipe = Pipeline([
                ('ordinal', OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=self.unknown_value
                ))
            ])
            transformers.append(('ordinal', ordinal_pipe, self.ordinal_features))
            logger.info(f"Added OrdinalEncoder for features: {self.ordinal_features}")

        # One-hot encoding
        if self.nominal_features:
            nominal_pipe = Pipeline([
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore' if self.handle_unknown == 'ignore' else 'error',
                    sparse_output=False
                ))
            ])
            transformers.append(('onehot', nominal_pipe, self.nominal_features))
            logger.info(f"Added OneHotEncoder for features: {self.nominal_features}")

        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            
            try:
                self.preprocessor.fit(X)
                self.feature_names_out_ = self.preprocessor.get_feature_names_out()
                logger.info("Fitted preprocessing pipeline (ColumnTransformer)")
            except Exception as e:
                raise ValueError(f"Error fitting ColumnTransformer: {str(e)}")
        else:
            self.preprocessor = None
            logger.info("No ColumnTransformer created as no nominal or ordinal features were identified")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """Fit all encoders to the training data.
        
        Args:
            X: DataFrame containing features to encode
            y: Optional target series (required for target encoding)
            
        Returns:
            self: Fitted encoder instance
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        if self.high_cardinality_strategy == 'target' and y is None:
            raise ValueError("y must be provided when using high_cardinality_strategy='target'")

        # Reset state in case fit is called multiple times
        self.label_encoders = {}
        self.bin_encoders = {}
        self.target_encoders = {}
        self.preprocessor = None
        self.feature_names_out_ = None

        # Identify feature types
        self._identify_feature_types(X)

        # Fit binary encoding
        if self.binary_features:
            self._fit_binary_encoding(X)

        # Fit high cardinality encoding
        if self.high_cardinality_features:
            self._fit_high_cardinality_encoding(X, y)

        # Fit nominal/ordinal encoding
        self._fit_nominal_ordinal_encoding(X)

        return self

    def _transform_binary_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply binary feature encoding to new data."""
        X_transformed = X.copy()
        
        for col in self.binary_features:
            if col not in X_transformed.columns:
                logger.warning(f"Binary feature '{col}' not found in input data")
                continue
                
            try:
                # Convert to string and handle unseen values
                encoded = X_transformed[col].astype(str)
                unseen_mask = ~encoded.isin(self.label_encoders[col].classes_)
                
                if unseen_mask.any():
                    if self.handle_unknown == 'error':
                        raise ValueError(
                            f"Found unknown categories in binary feature '{col}': "
                            f"{encoded[unseen_mask].unique().tolist()}"
                        )
                    elif self.handle_unknown == 'impute':
                        logger.warning(
                            f"Imputing {unseen_mask.sum()} unknown values in binary feature '{col}' "
                            f"with {self.unknown_value}"
                        )
                        encoded[unseen_mask] = self.unknown_value
                
                X_transformed[col] = self.label_encoders[col].transform(encoded)
            except Exception as e:
                raise ValueError(f"Error transforming binary feature '{col}': {str(e)}")
                
        return X_transformed

    def _transform_high_cardinality_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply high cardinality encoding to new data."""
        if not self.high_cardinality_features:
            return X
            
        X_transformed = X.copy()
        
        for col in self.high_cardinality_features:
            if col not in X_transformed.columns:
                logger.warning(f"High-cardinality feature '{col}' not found in input data")
                continue
                
            if self.high_cardinality_strategy == 'frequency' and col in self.bin_encoders:
                try:
                    # KBins expects 2D input and returns 2D output
                    X_transformed[col] = self.bin_encoders[col].transform(X_transformed[[col]]).flatten()
                except Exception as e:
                    raise ValueError(
                        f"Error applying frequency encoding to high-cardinality feature '{col}': {str(e)}"
                    )
                    
            elif self.high_cardinality_strategy == 'clustering' and col in self.bin_encoders:
                try:
                    le, kmeans = self.bin_encoders[col]
                    encoded = le.transform(X_transformed[col].astype(str)).reshape(-1, 1)
                    X_transformed[col] = kmeans.predict(encoded)
                except Exception as e:
                    raise ValueError(
                        f"Error applying cluster encoding to high-cardinality feature '{col}': {str(e)}"
                    )
                    
            elif self.high_cardinality_strategy == 'target' and col in self.target_encoders:
                try:
                    target_map = self.target_encoders[col]
                    default_value = np.mean(list(target_map.values())) if target_map else self.unknown_value
                    X_transformed[col] = X_transformed[col].map(target_map).fillna(default_value)
                except Exception as e:
                    raise ValueError(
                        f"Error applying target encoding to high-cardinality feature '{col}': {str(e)}"
                    )
        
        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all learned encodings to new data.
        
        Args:
            X: DataFrame containing features to encode
            
        Returns:
            DataFrame: Transformed data with encoded features
            
        Raises:
            NotFittedError: If encoder has not been fitted
            ValueError: If input data validation fails
        """
        if not hasattr(self, 'binary_features'):
            raise NotFittedError("CategoricalEncoder has not been fitted yet")
            
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        logger.info("Starting transformation of input data")

        # Apply binary encoding
        X_transformed = self._transform_binary_features(X)

        # Apply high cardinality encoding
        X_transformed = self._transform_high_cardinality_features(X_transformed)

        # Apply nominal/ordinal encoding via ColumnTransformer
        if self.preprocessor is not None:
            try:
                transformed_array = self.preprocessor.transform(X_transformed)
                feature_names = self.preprocessor.get_feature_names_out()
                X_transformed = pd.DataFrame(
                    transformed_array,
                    index=X_transformed.index,
                    columns=feature_names
                )
            except Exception as e:
                raise ValueError(f"Error in ColumnTransformer transformation: {str(e)}")

        # Ensure numerical features are retained if they weren't processed
        for col in self.numerical_features:
            if col in X.columns and col not in X_transformed.columns:
                X_transformed[col] = X[col]

        logger.info("Successfully completed data transformation")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation.
        
        Returns:
            List of feature names
            
        Raises:
            NotFittedError: If encoder has not been fitted
        """
        if not hasattr(self, 'binary_features'):
            raise NotFittedError("CategoricalEncoder has not been fitted yet")
            
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
            
        # Fallback when no ColumnTransformer was used
        feature_names = []
        
        # Binary features
        feature_names.extend(self.binary_features)
        
        # High cardinality features
        feature_names.extend(self.high_cardinality_features)
        
        # Numerical features
        feature_names.extend(self.numerical_features)
        
        return feature_names

    def save(self, filepath: str) -> None:
        """Save encoder state to file.
        
        Args:
            filepath: Path to save the encoder
            
        Raises:
            ValueError: If filepath is invalid
            IOError: If saving fails
        """
        if not filepath:
            raise ValueError("filepath cannot be empty")
            
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'preprocessor': self.preprocessor,
                    'label_encoders': self.label_encoders,
                    'bin_encoders': self.bin_encoders,
                    'target_encoders': self.target_encoders,
                    'feature_names_out_': self.feature_names_out_,
                    'feature_types': {
                        'binary': self.binary_features,
                        'nominal': self.nominal_features,
                        'ordinal': self.ordinal_features,
                        'high_cardinality': self.high_cardinality_features,
                        'numerical': self.numerical_features
                    },
                    'params': {
                        'column_encoding_strategy': self.column_encoding_strategy,
                        'ordinal_features_map': self.ordinal_features_map,
                        'max_cardinality': self.max_cardinality,
                        'high_cardinality_strategy': self.high_cardinality_strategy,
                        'handle_unknown': self.handle_unknown,
                        'unknown_value': self.unknown_value,
                        'target_col': self.target_col,
                        'random_state': self.random_state
                    },
                    'sklearn_version': sklearn.__version__
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Successfully saved CategoricalEncoder to {filepath}")
        except Exception as e:
            raise IOError(f"Error saving CategoricalEncoder to {filepath}: {str(e)}")

    @classmethod
    def load(cls, filepath: str) -> 'CategoricalEncoder':
        """Load encoder from file.
        
        Args:
            filepath: Path to load the encoder from
            
        Returns:
            CategoricalEncoder: Loaded encoder instance
            
        Raises:
            ValueError: If filepath is invalid
            IOError: If loading fails
        """
        if not filepath:
            raise ValueError("filepath cannot be empty")
            
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
        except Exception as e:
            raise IOError(f"Error loading CategoricalEncoder from {filepath}: {str(e)}")

        # Version compatibility check
        current_sklearn_version = sklearn.__version__
        saved_sklearn_version = state.get('sklearn_version', 'unknown')
        if saved_sklearn_version != 'unknown' and current_sklearn_version != saved_sklearn_version:
            logger.warning(
                f"Loading encoder trained with sklearn version {saved_sklearn_version}, "
                f"but current version is {current_sklearn_version}. "
                "Compatibility issues may arise with scikit-learn models."
            )

        # Create new instance with saved parameters
        encoder = cls(**state['params'])
        
        # Restore state
        encoder.preprocessor = state['preprocessor']
        encoder.label_encoders = state.get('label_encoders', {})
        encoder.bin_encoders = state.get('bin_encoders', {})
        encoder.target_encoders = state.get('target_encoders', {})
        encoder.feature_names_out_ = state.get('feature_names_out_')
        
        # Restore feature types
        feature_types = state.get('feature_types', {})
        encoder.binary_features = feature_types.get('binary', [])
        encoder.nominal_features = feature_types.get('nominal', [])
        encoder.ordinal_features = feature_types.get('ordinal', [])
        encoder.high_cardinality_features = feature_types.get('high_cardinality', [])
        encoder.numerical_features = feature_types.get('numerical', [])
        
        logger.info(f"Successfully loaded CategoricalEncoder from {filepath}")
        return encoder

    def get_feature_type_summary(self) -> Dict[str, List[str]]:
        """Get a summary of feature types identified during fitting.
        
        Returns:
            Dictionary mapping feature types to column names
        """
        return {
            'binary': self.binary_features,
            'nominal': self.nominal_features,
            'ordinal': self.ordinal_features,
            'high_cardinality': self.high_cardinality_features,
            'numerical': self.numerical_features
        }