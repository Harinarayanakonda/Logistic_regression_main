import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from src.utils.logging import logger

class FeatureExtractor:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = None
        logger.info(f"FeatureExtractor initialized with n_components={n_components}")

    def fit(self, X: pd.DataFrame):
        numeric_cols = X.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < self.n_components:
            raise ValueError("Not enough numeric columns for PCA.")
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X[numeric_cols].dropna())
        self.numeric_cols = numeric_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.pca is None:
            raise RuntimeError("FeatureExtractor not fitted.")
        principal_components = self.pca.transform(X_copy[self.numeric_cols].fillna(0))
        for i in range(self.n_components):
            X_copy[f'PC{i+1}'] = principal_components[:, i]
        return X_copy

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

class FeatureCreator:
    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        numeric_cols = X_copy.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            X_copy['synthetic_sum'] = X_copy[numeric_cols].sum(axis=1)
        return X_copy

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.transform(X)

class FeatureSelector:
    def __init__(self, k=5):
        self.k = k
        self.selector = None
        self.selected_features = None
        logger.info(f"FeatureSelector initialized with k={k}")

    def fit(self, X: pd.DataFrame, y):
        numeric_cols = X.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < self.k:
            raise ValueError("Not enough numeric columns for selection.")
        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        self.selector.fit(X[numeric_cols], y)
        self.selected_features = [numeric_cols[i] for i in self.selector.get_support(indices=True)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features is None:
            raise RuntimeError("FeatureSelector not fitted.")
        X_copy = X.copy()
        return X_copy[self.selected_features + (['target'] if 'target' in X_copy.columns else [])]

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)