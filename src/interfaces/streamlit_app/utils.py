import streamlit as st
import pandas as pd
import os
import joblib
from io import StringIO, BytesIO
from src.core.data_loader import DataLoader
from src.utils.logging import logger
from src.config.paths import PREPROCESSING_ARTIFACTS_DIR, TRAINED_MODELS_DIR
from src.utils.helpers import save_artifact, load_artifact, generate_timestamped_filename
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import skew, zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

def load_data_from_uploader(uploaded_file, file_format) -> pd.DataFrame:
    """Helper to load data from Streamlit's uploaded file object."""
    data_loader = DataLoader()
    try:
        if uploaded_file is not None:
            if file_format in ['csv', 'txt']:
                df = data_loader.load_data(StringIO(uploaded_file.getvalue().decode('utf-8')))
            elif file_format in ['xlsx', 'xls', 'xlsm']:
                df = data_loader.load_data(BytesIO(uploaded_file.getvalue()))
            elif file_format == 'json':
                df = data_loader.load_data(StringIO(uploaded_file.getvalue().decode('utf-8')))
            else:
                st.error("Unsupported file format selected.")
                return None
            return df
        return None
    except Exception as e:
        logger.error(f"Error loading uploaded file: {e}")
        st.error(f"Failed to load dataset: {e}. Please ensure the file format is correct and the file is not corrupted.")
        return None

def save_preprocessing_artifacts(preprocessing_pipeline, model_object, target_column):
    """Saves the preprocessing pipeline and model to a timestamped file."""
    timestamped_filename = generate_timestamped_filename("model_artifacts")
    filepath = os.path.join(TRAINED_MODELS_DIR, timestamped_filename)
    artifact_to_save = {
        'preprocessing_pipeline': preprocessing_pipeline,
        'model': model_object,
        'target_column': target_column,
        'fitted_features': preprocessing_pipeline.get_feature_names_after_preprocessing()
    }
    try:
        save_artifact(artifact_to_save, filepath)
        st.session_state['last_saved_model_path'] = filepath
        st.success(f"💾 Preprocessed pipeline and model saved as **{os.path.basename(filepath)}**")
        logger.info(f"Saved combined model artifact to: {filepath}")
        return True
    except Exception as e:
        st.error(f"Failed to save preprocessing artifacts and model: {e}")
        logger.error(f"Error saving combined model artifact: {e}")
        return False

def load_latest_model_artifact():
    """Loads the latest saved model artifact."""
    model_files = [f for f in os.listdir(TRAINED_MODELS_DIR) if f.startswith('model_artifacts_') and f.endswith('.pkl')]
    if not model_files:
        logger.warning("No model artifacts found.")
        return None
    model_files.sort(reverse=True)
    latest_model_path = os.path.join(TRAINED_MODELS_DIR, model_files[0])
    try:
        artifact = load_artifact(latest_model_path)
        logger.info(f"Loaded latest model artifact: {latest_model_path}")
        return artifact
    except Exception as e:
        logger.error(f"Error loading latest model artifact from {latest_model_path}: {e}")
        return None

def download_dataframe(df: pd.DataFrame, file_name: str, file_format: str):
    """Provides a download button for a DataFrame."""
    if file_format == 'csv':
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{file_name}.csv",
            mime="text/csv",
        )
    elif file_format == 'pkl':
        pkl = joblib.dumps(df)
        st.download_button(
            label="Download Data as PKL",
            data=pkl,
            file_name=f"{file_name}.pkl",
            mime="application/octet-stream",
        )
    else:
        st.error("Unsupported file format for download.")

# --- Intelligent Imputation Utility ---
def intelligent_impute(
    df: pd.DataFrame,
    is_time_series: bool = False,
    time_col: str = None,
    critical_cols: list = None,
    corr_threshold: float = 0.5,
    knn_neighbors: int = 5,
    random_state: int = 42
):
    """
    Intelligent missing value handling for any dataset.
    Returns: cleaned_df, summary_table, imputation_log
    """
    df = df.copy()
    imputation_log = []
    summary = []

    if critical_cols is None:
        critical_cols = []

    # 1. Flag missing values and calculate % missing
    missing_info = df.isnull().mean() * 100
    missing_cols = missing_info[missing_info > 0].index.tolist()

    # 2. For time series, sort by time_col if provided
    if is_time_series and time_col and time_col in df.columns:
        df = df.sort_values(by=time_col)

    # 3. For each column, decide and apply imputation
    for col in missing_cols:
        pct_missing = missing_info[col]
        col_type = df[col].dtype
        n_missing = df[col].isnull().sum()
        chosen_method = None

        # For time series
        if is_time_series:
            if col_type in [np.float64, np.int64]:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                chosen_method = "forward/backward fill"
            else:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                chosen_method = "forward/backward fill"
            imputation_log.append(f"{col}: {chosen_method} (time series)")
            summary.append([col, pct_missing, chosen_method])
            continue

        # < 5% missing
        if pct_missing < 5:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check skewness
                col_skew = skew(df[col].dropna())
                if abs(col_skew) < 1:
                    df[col] = df[col].fillna(df[col].mean())
                    chosen_method = "mean"
                else:
                    df[col] = df[col].fillna(df[col].median())
                    chosen_method = "median"
            else:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                df[col] = df[col].fillna(mode_val)
                chosen_method = "mode"
            imputation_log.append(f"{col}: {chosen_method} imputation (<5% missing)")
        
        # 5-30% missing
        elif 5 <= pct_missing <= 30:
            # Try KNN imputation for numeric/categorical
            if pd.api.types.is_numeric_dtype(df[col]):
                knn_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                df[knn_cols] = imputer.fit_transform(df[knn_cols])
                chosen_method = "KNN imputation"
            else:
                # For categorical, use KNN on encoded data
                df_cat = df.select_dtypes(include=['object', 'category']).copy()
                df_cat_enc = df_cat.apply(lambda x: pd.factorize(x)[0])
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                df_cat_imputed = imputer.fit_transform(df_cat_enc)
                for i, c in enumerate(df_cat.columns):
                    labels = list(df_cat[c].dropna().unique())
                    df[c] = [labels[int(idx)] if not np.isnan(idx) and int(idx) < len(labels) else np.nan for idx in df_cat_imputed[:, i]]
                chosen_method = "KNN imputation (categorical)"
            imputation_log.append(f"{col}: {chosen_method} (5-30% missing)")
        
        # > 30% missing
        else:
            # Check if critical or highly correlated
            if col in critical_cols:
                # Try model-based imputation if possible
                # Use correlation to find predictors
                corr = df.corr(numeric_only=True)
                if col in corr.columns:
                    predictors = corr[col].abs().sort_values(ascending=False)
                    predictors = predictors[predictors.index != col]
                    strong_predictors = predictors[predictors > corr_threshold].index.tolist()
                else:
                    strong_predictors = []
                if strong_predictors:
                    # Simple model-based imputation (linear regression for numeric)
                    from sklearn.linear_model import LinearRegression
                    not_null = df[df[col].notnull()]
                    null = df[df[col].isnull()]
                    if not_null.shape[0] > 0 and len(strong_predictors) > 0:
                        lr = LinearRegression()
                        lr.fit(not_null[strong_predictors], not_null[col])
                        df.loc[df[col].isnull(), col] = lr.predict(null[strong_predictors])
                        chosen_method = f"model-based imputation using {strong_predictors}"
                    else:
                        df = df.drop(columns=[col])
                        chosen_method = "dropped (not enough data for model-based imputation)"
                else:
                    df = df.drop(columns=[col])
                    chosen_method = "dropped (>30% missing, not critical or correlated)"
            else:
                df = df.drop(columns=[col])
                chosen_method = "dropped (>30% missing, not critical)"
            imputation_log.append(f"{col}: {chosen_method}")
        
        summary.append([col, pct_missing, chosen_method])

    # 4. Summary table
    summary_table = pd.DataFrame(summary, columns=["Column", "% Missing", "Imputation Method"])

    return df, summary_table, imputation_log

# --- Intelligent Outlier Handler Utility ---
def intelligent_outlier_handler(
    df: pd.DataFrame,
    visualize: bool = False,
    random_state: int = 42
):
    """
    Intelligent outlier detection and handling for any dataset.
    Returns: cleaned_df, summary_table, outlier_flags
    """
    df = df.copy()
    summary = []
    cleaned_df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_flags = pd.DataFrame(index=df.index)
    
    for col in num_cols:
        col_data = df[col].dropna()
        outlier_idx = set()
        method_used = []
        
        # IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)].index
        outlier_idx.update(iqr_outliers)
        method_used.append("IQR")
        
        # Z-score method (only if approx normal)
        if abs(col_data.skew()) < 1:
            z_scores = zscore(col_data)
            z_outliers = col_data[(np.abs(z_scores) > 3)].index
            outlier_idx.update(z_outliers)
            method_used.append("Z-score")
        
        # Isolation Forest (for multivariate/complex)
        if len(col_data) > 50:
            iso = IsolationForest(contamination=0.05, random_state=random_state)
            preds = iso.fit_predict(col_data.values.reshape(-1, 1))
            iso_outliers = col_data.index[preds == -1]
            outlier_idx.update(iso_outliers)
            method_used.append("IsolationForest")
        
        outlier_count = len(outlier_idx)
        pct_outliers = 100 * outlier_count / len(col_data)
        chosen_method = ""
        
        # Handling rules
        if pct_outliers < 5:
            cleaned_df = cleaned_df.drop(index=outlier_idx)
            chosen_method = "Remove rows"
        elif 5 <= pct_outliers <= 15:
            # Winsorization (cap)
            lower = col_data.quantile(0.01)
            upper = col_data.quantile(0.99)
            cleaned_df[col] = np.clip(cleaned_df[col], lower, upper)
            chosen_method = "Winsorization"
            # Log if right-skewed
            if col_data.skew() > 1:
                cleaned_df[col] = np.log1p(cleaned_df[col])
                chosen_method += " + Log Transform"
        else:
            # Robust scaling
            scaler = RobustScaler()
            cleaned_df[col] = scaler.fit_transform(cleaned_df[[col]])
            chosen_method = "RobustScaler"
        
        summary.append({
            "Feature": col,
            "Outlier Count": outlier_count,
            "% Outliers": round(pct_outliers, 2),
            "Detection Methods": ", ".join(method_used),
            "Handling Method": chosen_method
        })
        
        # For visualization
        outlier_flags[col] = df.index.isin(outlier_idx).astype(int)
        
        if visualize:
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.boxplot(y=df[col], ax=axes[0])
            axes[0].set_title(f"{col} - Before")
            sns.boxplot(y=cleaned_df[col], ax=axes[1])
            axes[1].set_title(f"{col} - After")
            st.pyplot(fig)
            plt.close(fig)
    
    summary_table = pd.DataFrame(summary)
    return cleaned_df, summary_table, outlier_flags

def intelligent_log_transform(df: pd.DataFrame):
    """
    Automatically detects right-skewed numerical features and applies the best log/Box-Cox/Yeo-Johnson transformation.
    Returns:
        - transformed DataFrame
        - summary table (feature, original/new skewness, method)
        - log of transformations
    """
    df = df.copy()
    log = []
    summary = []
    transformed_features = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        col_data = df[col].dropna()
        orig_skew = skew(col_data)
        method = None
        new_skew = orig_skew
        # Only transform if right-skewed
        if orig_skew > 0.75:
            # If all values > 0
            if (col_data > 0).all():
                # If zeros present, use log1p
                if (col_data == 0).any():
                    df[col] = np.log1p(df[col])
                    method = "log1p"
                else:
                    # Try Box-Cox (more flexible)
                    try:
                        df[col], _ = boxcox(df[col])
                        method = "Box-Cox"
                    except Exception:
                        df[col] = np.log(df[col])
                        method = "log"
            # If values include zero or negative, use Yeo-Johnson
            elif (col_data >= 0).all():
                pt = PowerTransformer(method='yeo-johnson')
                df[col] = pt.fit_transform(df[[col]])
                method = "Yeo-Johnson"
            else:
                # Contains negatives, use Yeo-Johnson
                pt = PowerTransformer(method='yeo-johnson')
                df[col] = pt.fit_transform(df[[col]])
                method = "Yeo-Johnson"
            # Check new skewness
            new_skew = skew(df[col].dropna())
            transformed_features.append(col)
            log.append(f"{col}: {method} (skew {orig_skew:.2f} → {new_skew:.2f})")
        summary.append({
            "Feature": col,
            "Original Skewness": round(orig_skew, 3),
            "Transformed?": "Yes" if method else "No",
            "Method": method if method else "-",
            "New Skewness": round(new_skew, 3)
        })
        # Ensure no NaNs or infs
        if df[col].isnull().any() or np.isinf(df[col]).any():
            raise ValueError(f"NaN or inf detected in {col} after {method} transformation.")
    summary_table = pd.DataFrame(summary)
    return df, summary_table, log

def intelligent_scaling(
    df: pd.DataFrame,
    model_type: str = "general",  # "tree" or "general"
    outlier_threshold: float = 3.5
):
    """
    Automatically applies the best scaling technique to each numerical feature.
    Returns:
        - scaled DataFrame
        - summary table (feature, distribution, scaler used)
        - dict of fitted scaler objects
    """
    df = df.copy()
    summary = []
    scalers = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        col_data = df[col].dropna()
        # Detect outliers
        iqr = col_data.quantile(0.75) - col_data.quantile(0.25)
        lower = col_data.quantile(0.25) - 1.5 * iqr
        upper = col_data.quantile(0.75) + 1.5 * iqr
        outlier_frac = ((col_data < lower) | (col_data > upper)).mean()
        # Detect distribution
        try:
            stat, p = normaltest(col_data)
            is_normal = (p > 0.05)
        except Exception:
            is_normal = False
        col_skew = skew(col_data)
        scaler_used = None
        distribution = "normal" if is_normal else "skewed"
        # Decision logic
        if outlier_frac > 0.05:
            scaler = RobustScaler()
            scaler_used = "RobustScaler"
        elif is_normal:
            scaler = StandardScaler()
            scaler_used = "StandardScaler"
        elif (col_data.min() >= 0) and (col_data.max() <= 1):
            scaler = MinMaxScaler()
            scaler_used = "MinMaxScaler"
        else:
            scaler = MinMaxScaler()
            scaler_used = "MinMaxScaler"
        # Tree-based models: scaling optional
        if model_type == "tree":
            scaler = None
            scaler_used = "None (tree model)"
        # Actually apply scaler if not tree
        if scaler is not None:
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler
        summary.append({
            "Feature": col,
            "Distribution": distribution,
            "Skewness": round(col_skew, 3),
            "Outlier Fraction": round(outlier_frac, 3),
            "Scaler Used": scaler_used
        })
        # Ensure no NaNs or infs
        if df[col].isnull().any() or np.isinf(df[col]).any():
            raise ValueError(f"NaN or inf detected in {col} after {scaler_used}.")
    summary_table = pd.DataFrame(summary)
    return df, summary_table, scalers

def intelligent_categorical_encoding(df: pd.DataFrame, target=None, cardinality_threshold=10):
    df = df.copy()
    summary = []
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        n_unique = df[col].nunique()
        if pd.api.types.is_categorical_dtype(df[col]) and df[col].cat.ordered:
            # Ordinal encoding
            encoder = OrdinalEncoder()
            df[col] = encoder.fit_transform(df[[col]])
            method = "Ordinal"
        elif n_unique <= cardinality_threshold:
            # One-hot encoding
            ohe = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), ohe], axis=1)
            method = "One-hot"
        else:
            # Frequency encoding for high cardinality
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
            method = "Frequency"
        summary.append({"Feature": col, "Unique": n_unique, "Encoding": method})
    return df, pd.DataFrame(summary)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def reduce_multicollinearity(df: pd.DataFrame, vif_thresh=10, target=None, critical_cols=None, corr_thresh=0.9):
    df = df.copy()
    summary = []
    if critical_cols is None:
        critical_cols = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Drop target from VIF calculation
    if target in num_cols:
        num_cols.remove(target)
    dropped = []
    while True:
        X = df[num_cols].dropna()
        vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
        max_vif = vif.max()
        if max_vif > vif_thresh:
            drop_col = vif.idxmax()
            if drop_col not in critical_cols:
                df = df.drop(columns=[drop_col])
                num_cols.remove(drop_col)
                dropped.append(drop_col)
                summary.append({"Feature": drop_col, "VIF": max_vif, "Reason": "High VIF"})
            else:
                break
        else:
            break
    # Drop one of two highly correlated features
    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > corr_thresh and col not in critical_cols and row not in critical_cols:
                df = df.drop(columns=[col])
                num_cols.remove(col)
                dropped.append(col)
                summary.append({"Feature": col, "VIF": "-", "Reason": f"High Corr with {row}"})
    return df, pd.DataFrame(summary)
from sklearn.preprocessing import PolynomialFeatures

def intelligent_feature_engineering(df: pd.DataFrame, degree=2, datetime_cols=None, binning_cols=None):
    df = df.copy()
    summary = []
    # Polynomial features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[num_cols])
        poly_feature_names = poly.get_feature_names_out(num_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        # Only add new features (not original)
        for col in poly_df.columns:
            if col not in df.columns:
                df[col] = poly_df[col]
                summary.append({"Feature": col, "Type": "Polynomial"})
    # Interaction terms (already included above if degree > 1)
    # Datetime features
    if datetime_cols:
        for col in datetime_cols:
            df[f"{col}_day"] = pd.to_datetime(df[col]).dt.day
            df[f"{col}_month"] = pd.to_datetime(df[col]).dt.month
            df[f"{col}_weekday"] = pd.to_datetime(df[col]).dt.weekday
            summary.extend([
                {"Feature": f"{col}_day", "Type": "Datetime"},
                {"Feature": f"{col}_month", "Type": "Datetime"},
                {"Feature": f"{col}_weekday", "Type": "Datetime"},
            ])
    # Binning
    if binning_cols:
        for col, bins in binning_cols.items():
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, labels=False)
            summary.append({"Feature": f"{col}_bin", "Type": "Binned"})
    return df, pd.DataFrame(summary)