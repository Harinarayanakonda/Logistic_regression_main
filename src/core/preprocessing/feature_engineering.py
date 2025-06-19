# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, List, Dict, Any, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ---------- Named Functions for Feature Creation ----------
def sum_features(df):
    return df.sum(axis=1) if len(df.select_dtypes(include=np.number).columns) > 1 else None

def mean_features(df):
    return df.mean(axis=1) if len(df.select_dtypes(include=np.number).columns) > 1 else None

def product_features(df):
    return df.product(axis=1) if len(df.select_dtypes(include=np.number).columns) > 1 else None

def log_sum(df):
    return np.log1p(df.sum(axis=1)) if len(df.select_dtypes(include=np.number).columns) > 1 else None

def max_feature(df):
    return df.max(axis=1) if len(df.select_dtypes(include=np.number).columns) > 1 else None

def min_feature(df):
    return df.min(axis=1) if len(df.select_dtypes(include=np.number).columns) > 1 else None

class BaseFeatureEngineer(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for all feature engineering components."""
    
    def __init__(self):
        self.feature_names_out_ = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        if self.feature_names_out_ is None:
            raise ValueError("Feature names are not available before fitting the transformer.")
        return self.feature_names_out_


class FeatureExtractor(BaseFeatureEngineer):
    """
    Feature extraction using PCA with automatic naming of components.
    
    Parameters:
    -----------
    n_components : int, float or str, default=None
        Number of components to keep. If None, all components are kept.
        If 0 < n_components < 1, select the number of components such that the
        amount of variance that needs to be explained is greater than the percentage specified.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_components: Optional[Union[int, float]] = None, random_state: Optional[int] = None):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
        self.extracted_feature_names = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit PCA to the data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
        
        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for PCA")
            
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X[numeric_cols])
        
        # Generate feature names
        n_components = self.pca.n_components_
        self.extracted_feature_names = [f"PC_{i+1}" for i in range(n_components)]
        self.feature_names_out_ = self.extracted_feature_names
        
        logger.info(f"PCA fitted with {n_components} components explaining {self.pca.explained_variance_ratio_.sum():.2%} of variance")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted PCA."""
        if self.pca is None:
            raise RuntimeError("PCA must be fitted before transformation")
            
        numeric_cols = X.select_dtypes(include=np.number).columns
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        
        transformed = self.pca.transform(X[numeric_cols])
        transformed_df = pd.DataFrame(transformed, columns=self.extracted_feature_names, index=X.index)
        
        # Preserve non-numeric columns
        if non_numeric_cols:
            return pd.concat([transformed_df, X[non_numeric_cols]], axis=1)
        return transformed_df
    
    def get_explained_variance(self) -> np.ndarray:
        """Return explained variance ratio of components."""
        if self.pca is None:
            raise RuntimeError("PCA must be fitted first")
        return self.pca.explained_variance_ratio_


class FeatureCreator(BaseFeatureEngineer):
    """
    Create new features based on existing ones with automatic feature naming.
    
    Parameters:
    -----------
    creation_rules : Optional[Dict[str, callable]], default=None
        Dictionary of feature creation rules where keys are new feature names
        and values are functions that take a DataFrame and return a Series.
    """
    
    def __init__(self, creation_rules: Optional[Dict[str, callable]] = None):
        super().__init__()
        self.creation_rules = creation_rules or self._get_default_creation_rules()
        self.new_features_added = False
        self.new_feature_names = []
    
    def _get_default_creation_rules(self) -> Dict[str, callable]:
        """Get default feature creation rules."""
        return {
            'sum_features': sum_features,
            'mean_features': mean_features,
            'product_features': product_features,
            'log_sum': log_sum,
            'max_feature': max_feature,
            'min_feature': min_feature
        }

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        df_transformed = self.transform(df)
        new_features = list(set(df_transformed.columns) - set(df.columns))
        return df_transformed, new_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Analyze data to determine which features can be created."""
        self.new_feature_names = []
        numeric_cols = X.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Need at least 2 numeric columns for feature creation")
            self.feature_names_out_ = list(X.columns)
            return self
        
        # Validate creation rules
        for name, func in self.creation_rules.items():
            try:
                sample_result = func(X[numeric_cols].head(2))
                if sample_result is not None:
                    self.new_feature_names.append(name)
            except Exception as e:
                logger.warning(f"Feature creation rule '{name}' failed: {e}")
                
        self.feature_names_out_ = list(X.columns) + self.new_feature_names
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature creation rules to the data."""
        X_transformed = X.copy()
        numeric_cols = X.select_dtypes(include=np.number).columns
        self.new_features_added = False
        
        if len(numeric_cols) < 2:
            return X_transformed
            
        for name in self.new_feature_names:
            try:
                new_feature = self.creation_rules[name](X_transformed[numeric_cols])
                if new_feature is not None:
                    X_transformed[name] = new_feature
                    self.new_features_added = True
            except Exception as e:
                logger.error(f"Failed to create feature '{name}': {e}")
                    
        if not self.new_features_added:
            logger.info("No new features were created - check input data and creation rules")
            
        return X_transformed
    
    def add_creation_rule(self, name: str, rule: callable):
        """Add a new feature creation rule."""
        if not callable(rule):
            raise ValueError("Rule must be callable")
        self.creation_rules[name] = rule
        logger.info(f"Added new feature creation rule: {name}")


class FeatureSelector(BaseFeatureEngineer):
    """
    Feature selection with multiple strategies including variance threshold,
    univariate selection, and correlation-based selection.
    
    Parameters:
    -----------
    strategy : str, default='univariate'
        Feature selection strategy. Options: 'variance', 'univariate', 'correlation'.
    threshold : float, default=0.0
        Threshold for variance or correlation selection.
    k : int or 'all', default='all'
        Number of top features to select for univariate selection.
    score_func : callable, default=f_classif
        Scoring function for univariate selection.
    """
    
    def __init__(self, strategy: str = 'univariate', threshold: float = 0.0,
                 k: Union[int, str] = 'all', score_func: callable = f_classif):
        super().__init__()
        self.strategy = strategy
        self.threshold = threshold
        self.k = k
        self.score_func = score_func
        self.selector = None
        self.selected_features_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature selector to the data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
            
        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for feature selection")
            
        if self.strategy == 'variance':
            self.selector = VarianceThreshold(threshold=self.threshold)
        elif self.strategy == 'univariate':
            if y is None:
                raise ValueError("Target y is required for univariate feature selection")
            self.selector = SelectKBest(score_func=self.score_func, k=self.k)
        elif self.strategy == 'correlation':
            if y is None:
                raise ValueError("Target y is required for correlation-based feature selection")
            self.selector = self._get_correlation_selector()
        else:
            raise ValueError(f"Unknown feature selection strategy: {self.strategy}")
            
        self.selector.fit(X[numeric_cols], y)
        
        if hasattr(self.selector, 'get_support'):
            self.selected_features_ = numeric_cols[self.selector.get_support()]
        else:
            self.selected_features_ = numeric_cols
            
        self.feature_names_out_ = list(self.selected_features_)
        
        logger.info(f"Feature selection completed. Selected {len(self.selected_features_)} features "
                  f"out of {len(numeric_cols)} using strategy '{self.strategy}'")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features."""
        if self.selector is None:
            raise RuntimeError("Feature selector must be fitted before transformation")
            
        numeric_cols = X.select_dtypes(include=np.number).columns
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        
        if len(self.selected_features_) == 0:
            logger.warning("No features were selected - returning non-numeric columns only")
            return X[non_numeric_cols].copy() if non_numeric_cols else pd.DataFrame(index=X.index)
            
        # Ensure we only keep the selected features that exist in X
        available_features = [f for f in self.selected_features_ if f in X.columns]
        selected_df = X[available_features].copy()
        
        # Preserve non-numeric columns
        if non_numeric_cols:
            return pd.concat([selected_df, X[non_numeric_cols]], axis=1)
        return selected_df
    
    def _get_correlation_selector(self) -> 'FeatureSelector':
        """Create a correlation-based feature selector (custom implementation)."""
        class CorrelationSelector(BaseEstimator):
            def __init__(self, threshold=0.0):
                self.threshold = threshold
                self.selected_indices_ = None
                
            def fit(self, X, y):
                try:
                    correlations = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) 
                                          for i in range(X.shape[1])])
                    self.selected_indices_ = np.where(correlations > self.threshold)[0]
                except:
                    self.selected_indices_ = np.arange(X.shape[1])
                return self
                
            def transform(self, X):
                return X[:, self.selected_indices_]
                
            def get_support(self):
                mask = np.zeros(X.shape[1], dtype=bool)
                mask[self.selected_indices_] = True
                return mask
                
        return CorrelationSelector(threshold=self.threshold)
    
    def get_feature_scores(self) -> pd.Series:
        """Get feature scores if available (for univariate selection)."""
        if not hasattr(self.selector, 'scores_'):
            raise AttributeError("Feature scores are only available for univariate selection")
        return pd.Series(self.selector.scores_, index=self.feature_names_out_)


class FeatureEngineeringPipeline:
    """
    Pipeline for orchestrating multiple feature engineering steps.
    
    Parameters:
    -----------
    steps : List[Tuple[str, BaseFeatureEngineer]]
        List of (name, transformer) tuples to be executed in order.
    """
    
    def __init__(self, steps: List[Tuple[str, BaseFeatureEngineer]]):
        self.steps = steps
        self.fitted_steps = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit all steps in the pipeline."""
        current_X = X.copy()
        
        for name, step in self.steps:
            logger.info(f"Fitting feature engineering step: {name}")
            try:
                step.fit(current_X, y)
                self.fitted_steps[name] = step
                current_X = step.transform(current_X)
            except Exception as e:
                logger.error(f"Error in step {name}: {str(e)}")
                raise
                
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations in the pipeline."""
        if not self.fitted_steps:
            raise RuntimeError("Pipeline must be fitted before transformation")
            
        current_X = X.copy()
        
        for name, step in self.steps:
            if name not in self.fitted_steps:
                raise RuntimeError(f"Step {name} was not fitted")
            current_X = step.transform(current_X)
            
        return current_X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform all steps in the pipeline."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get final output feature names after all transformations."""
        if not self.fitted_steps:
            raise RuntimeError("Pipeline must be fitted first")
            
        # Return feature names from the last step
        return self.steps[-1][1].get_feature_names_out()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # Create pipeline
    pipeline = FeatureEngineeringPipeline([
        ('extraction', FeatureExtractor(n_components=2)),
        ('creation', FeatureCreator()),
        ('selection', FeatureSelector(strategy='univariate', k=2))
    ])
    
    # Fit and transform
    X = data.drop(columns=['target'])
    y = data['target']
    
    try:
        transformed_data = pipeline.fit_transform(X, y)
        
        print("\nOriginal shape:", X.shape)
        print("Transformed shape:", transformed_data.shape)
        print("Final features:", pipeline.get_feature_names_out())
        print("\nTransformed data sample:")
        print(transformed_data.head())
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")