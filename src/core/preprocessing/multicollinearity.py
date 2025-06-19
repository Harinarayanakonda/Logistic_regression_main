import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Optional, Dict, Union, Tuple
import logging
import pickle
import warnings
from pathlib import Path
from sklearn.exceptions import NotFittedError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MulticollinearityReducer:
    """
    Robust multicollinearity reducer with VIF and correlation analysis.
    
    Features:
    - VIF-based feature elimination with iterative approach
    - Optional correlation matrix analysis
    - Feature importance integration
    - Comprehensive error handling
    - Detailed logging and documentation
    - Serialization support
    - Input validation
    - Memory efficiency
    
    Parameters:
    -----------
    vif_threshold : float, default=5.0
        Threshold for Variance Inflation Factor (VIF) above which features will be removed
    max_iter : int, default=20
        Maximum number of iterations for VIF elimination process
    use_correlation : bool, default=False
        Whether to perform additional correlation-based elimination
    corr_threshold : float, default=0.8
        Absolute correlation threshold for correlation-based elimination
    handle_missing : str, default='drop'
        How to handle missing values ('drop', 'fill_mean', 'fill_median')
    verbose : bool, default=True
        Whether to log detailed processing information
    """
    
    def __init__(self, 
                 vif_threshold: float = 5.0,
                 max_iter: int = 20,
                 use_correlation: bool = False,
                 corr_threshold: float = 0.8,
                 handle_missing: str = 'drop',
                 verbose: bool = True):
        self._validate_init_params(vif_threshold, max_iter, use_correlation, 
                                 corr_threshold, handle_missing)
        
        self.vif_threshold = vif_threshold
        self.max_iter = max_iter
        self.use_correlation = use_correlation
        self.corr_threshold = corr_threshold
        self.handle_missing = handle_missing
        self.verbose = verbose
        self.selected_features_ = None
        self.eliminated_features_ = []
        self.vif_history_ = []
        self.is_fitted_ = False
        
        if self.verbose:
            logger.info("MulticollinearityReducer initialized with parameters: "
                       f"vif_threshold={vif_threshold}, max_iter={max_iter}, "
                       f"use_correlation={use_correlation}, corr_threshold={corr_threshold}, "
                       f"handle_missing={handle_missing}")
    
    def _validate_init_params(self, vif_threshold, max_iter, use_correlation, 
                            corr_threshold, handle_missing):
        """Validate initialization parameters"""
        if not isinstance(vif_threshold, (int, float)) or vif_threshold < 1:
            raise ValueError("vif_threshold must be a number >= 1")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(use_correlation, bool):
            raise ValueError("use_correlation must be boolean")
        if not 0 <= corr_threshold <= 1:
            raise ValueError("corr_threshold must be between 0 and 1")
        if handle_missing not in ['drop', 'fill_mean', 'fill_median']:
            raise ValueError("handle_missing must be one of: 'drop', 'fill_mean', 'fill_median'")
    
    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and select numeric features"""
        X = X.copy()
        
        # Handle missing values
        if self.handle_missing == 'drop':
            X = X.dropna()
            if X.empty:
                raise ValueError("DataFrame is empty after dropping NA values")
        elif self.handle_missing.startswith('fill_'):
            method = self.handle_missing.split('_')[1]
            X = X.fillna(X.mean() if method == 'mean' else X.median())
        
        # Select numeric features
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric features found in the DataFrame")
            
        return X[numeric_cols]
    
    def _calculate_vifs(self, df: pd.DataFrame, features: List[str]) -> Optional[pd.DataFrame]:
        """Calculate VIFs for the current feature set"""
        if len(features) < 2:
            if self.verbose:
                logger.warning("Need at least 2 features for VIF calculation")
            return None
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vif_data = pd.DataFrame({'feature': features})
                vif_data['VIF'] = [
                    variance_inflation_factor(df[features].values, i) 
                    for i in range(len(features))
                ]
            return vif_data.sort_values('VIF', ascending=False)
        except Exception as e:
            logger.error(f"VIF calculation failed: {str(e)}")
            raise RuntimeError(f"Failed to calculate VIFs: {str(e)}")
    
    def _select_feature_to_remove(self, 
                                vifs: pd.DataFrame,
                                feature_importance: Optional[Dict[str, float]]) -> str:
        """Select which feature to remove based on VIF and optionally feature importance"""
        if feature_importance:
            vifs['importance'] = vifs['feature'].map(feature_importance).fillna(0)
            return vifs.sort_values(['VIF', 'importance'], ascending=[False, True])['feature'].iloc[0]
        return vifs.iloc[0]['feature']
    
    def _handle_correlation(self, 
                          df: pd.DataFrame,
                          features: List[str],
                          feature_importance: Optional[Dict[str, float]]) -> List[str]:
        """Handle highly correlated features"""
        if len(features) < 2:
            return features
            
        corr_matrix = df[features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        
        for feature in features:
            if feature in to_drop:
                continue
                
            correlated = [col for col in features 
                         if col != feature and 
                         col not in to_drop and
                         corr_matrix.loc[feature, col] > self.corr_threshold]
            
            if correlated:
                candidates = [feature] + correlated
                if feature_importance:
                    importance = {f: feature_importance.get(f, 0) for f in candidates}
                    to_remove = min(importance.items(), key=lambda x: x[1])[0]
                else:
                    # Remove the feature with highest average correlation to others
                    avg_corr = corr_matrix.loc[candidates, candidates].mean(axis=1)
                    to_remove = avg_corr.idxmax()
                
                to_drop.add(to_remove)
                if self.verbose:
                    logger.info(f"Removed '{to_remove}' due to high correlation")
        
        return [f for f in features if f not in to_drop]
    
    def fit(self, 
           X: pd.DataFrame, 
           feature_importance: Optional[Dict[str, float]] = None) -> 'MulticollinearityReducer':
        """
        Fit the reducer to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        feature_importance : Optional[Dict[str, float]]
            Dictionary of feature importance scores (higher is more important)
            
        Returns:
        --------
        self : MulticollinearityReducer
            The fitted reducer
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
            
        if X.empty:
            raise ValueError("Input DataFrame is empty")
            
        try:
            X_processed = self._preprocess_data(X)
            features = X_processed.columns.tolist()
            remaining_features = features.copy()
            self.eliminated_features_ = []
            self.vif_history_ = []
            
            # VIF elimination iterations
            for iteration in range(self.max_iter):
                if self.verbose:
                    logger.info(f"Iteration {iteration + 1}/{self.max_iter}")
                
                vifs = self._calculate_vifs(X_processed, remaining_features)
                if vifs is None:
                    break
                
                self.vif_history_.append(vifs.set_index('feature')['VIF'].to_dict())
                max_vif = vifs['VIF'].max()
                
                if max_vif <= self.vif_threshold:
                    if self.verbose:
                        logger.info(f"All VIFs <= threshold ({self.vif_threshold}), stopping")
                    break
                
                feature_to_remove = self._select_feature_to_remove(vifs, feature_importance)
                remaining_features.remove(feature_to_remove)
                self.eliminated_features_.append(feature_to_remove)
                
                if self.verbose:
                    logger.info(f"Removed '{feature_to_remove}' (VIF: {max_vif:.2f})")
            
            # Optional correlation elimination
            if self.use_correlation and remaining_features:
                remaining_features = self._handle_correlation(
                    X_processed, 
                    remaining_features,
                    feature_importance
                )
            
            # Store results
            self.selected_features_ = remaining_features + [
                col for col in X.columns 
                if col not in features  # Keep non-numeric columns
            ]
            self.is_fitted_ = True
            
            if self.verbose:
                logger.info(f"Feature reduction completed. "
                          f"Original features: {len(features)}, "
                          f"Selected features: {len(remaining_features)}, "
                          f"Eliminated features: {len(self.eliminated_features_)}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by removing multicollinear features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features to transform
            
        Returns:
        --------
        pd.DataFrame
            Transformed features with multicollinear features removed
        """
        if not self.is_fitted_:
            raise NotFittedError("This MulticollinearityReducer instance is not fitted yet. "
                               "Call 'fit' with appropriate arguments first.")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
            
        existing_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = set(self.selected_features_) - set(existing_features)
        
        if missing_features:
            logger.warning(f"The following features were selected during fitting but "
                         f"are missing in transform: {missing_features}")
        
        return X[existing_features].copy()
    
    def fit_transform(self, 
                     X: pd.DataFrame,
                     feature_importance: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        feature_importance : Optional[Dict[str, float]]
            Dictionary of feature importance scores (higher is more important)
            
        Returns:
        --------
        pd.DataFrame
            Transformed features with multicollinear features removed
        """
        return self.fit(X, feature_importance).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of selected features.
        
        Returns:
        --------
        List[str]
            List of selected feature names
        """
        if not self.is_fitted_:
            raise NotFittedError("This MulticollinearityReducer instance is not fitted yet.")
        return self.selected_features_.copy()
    
    def get_eliminated_features(self) -> List[str]:
        """
        Get the names of eliminated features.
        
        Returns:
        --------
        List[str]
            List of eliminated feature names
        """
        if not self.is_fitted_:
            raise NotFittedError("This MulticollinearityReducer instance is not fitted yet.")
        return self.eliminated_features_.copy()
    def get_selected_features(self):
        return self.selected_features_
    
    def get_vif_history(self) -> List[Dict[str, float]]:
        """
        Get the history of VIF calculations during fitting.
        
        Returns:
        --------
        List[Dict[str, float]]
            List of dictionaries containing VIF values at each iteration
        """
        if not self.is_fitted_:
            raise NotFittedError("This MulticollinearityReducer instance is not fitted yet.")
        return self.vif_history_.copy()
    
    def summary(self) -> str:
        """
        Get a summary of the reduction process.
        
        Returns:
        --------
        str
            Formatted summary string
        """
        if not self.is_fitted_:
            return "MulticollinearityReducer not fitted yet."
            
        summary_lines = [
            "Multicollinearity Reduction Summary",
            "=" * 40,
            f"VIF Threshold: {self.vif_threshold}",
            f"Max Iterations: {self.max_iter}",
            f"Use Correlation: {self.use_correlation}",
            f"Correlation Threshold: {self.corr_threshold if self.use_correlation else 'N/A'}",
            f"Missing Value Handling: {self.handle_missing}",
            "",
            f"Original Features: {len(self.selected_features_ + self.eliminated_features_)}",
            f"Selected Features: {len(self.selected_features_)}",
            f"Eliminated Features: {len(self.eliminated_features_)}",
            ""
        ]
        
        if self.eliminated_features_:
            summary_lines.append("Eliminated Features:")
            summary_lines.extend([f"- {feat}" for feat in self.eliminated_features_])
        else:
            summary_lines.append("No features were eliminated.")
        
        return "\n".join(summary_lines)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the reducer to a file.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Path to save the reducer
        """
        if not self.is_fitted_:
            raise NotFittedError("Cannot save an unfitted reducer. Fit the reducer first.")
            
        state = {
            'selected_features': self.selected_features_,
            'eliminated_features': self.eliminated_features_,
            'vif_history': self.vif_history_,
            'params': {
                'vif_threshold': self.vif_threshold,
                'max_iter': self.max_iter,
                'use_correlation': self.use_correlation,
                'corr_threshold': self.corr_threshold,
                'handle_missing': self.handle_missing,
                'verbose': self.verbose
            },
            '_is_fitted': True
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        if self.verbose:
            logger.info(f"Saved reducer to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MulticollinearityReducer':
        """
        Load a reducer from a file.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Path to load the reducer from
            
        Returns:
        --------
        MulticollinearityReducer
            The loaded reducer instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        reducer = cls(**state['params'])
        reducer.selected_features_ = state['selected_features']
        reducer.eliminated_features_ = state['eliminated_features']
        reducer.vif_history_ = state['vif_history']
        reducer.is_fitted_ = state['_is_fitted']
        
        if reducer.verbose:
            logger.info(f"Loaded reducer from {filepath}")
        
        return reducer
    
    def __repr__(self) -> str:
        return (f"MulticollinearityReducer(vif_threshold={self.vif_threshold}, "
               f"max_iter={self.max_iter}, use_correlation={self.use_correlation}, "
               f"corr_threshold={self.corr_threshold}, handle_missing='{self.handle_missing}')")