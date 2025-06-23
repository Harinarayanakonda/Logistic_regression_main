import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import traceback
import json
# Add these missing imports at the top
from pathlib import Path
import joblib  # For pipeline serialization

import logging
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer  # For advanced imputation
from sklearn.decomposition import PCA 
from io import BytesIO
import json
from sklearn.impute import SimpleImputer


from sklearn.pipeline import Pipeline as SklearnPipeline
from src.core.preprocessing.log_transform import LogTransformer # For FeatureExtractor
from src.core.preprocessing.missing_values import MissingValueHandler
from src.core.preprocessing.outlier_detection import OutlierDetector, OutlierHandlingMethod, DetectionMethod 
from src.core.preprocessing.log_transform import LogTransformer
from src.core.preprocessing.scaling import FeatureScaler, ScalingMethod, ScalingResult
from src.core.preprocessing.multicollinearity import MulticollinearityReducer
from src.core.preprocessing.categorical_encoding import (
    CategoricalEncoder,
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    KBinsDiscretizer,
    ColumnTransformer,
    Pipeline,
    KMeans
)
from src.core.preprocessing.feature_engineering import (
    BaseFeatureEngineer,
    FeatureExtractor,
    FeatureCreator,
    FeatureSelector,
    FeatureEngineeringPipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dummy classes for demonstration. Replace with your actual implementations.
class AppSettings:
    TARGET_COLUMN = "target"  # Example target column
    OUTLIER_IQ_FACTOR = 1.5
    OUTLIER_ZSCORE_FACTOR = 3.0
    ORDINAL_FEATURES_MAP = {}  # Example, replace with actual map if needed

# Utility functions
def separate_features_and_target(df: pd.DataFrame, target_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[[target_col]].copy()
    else:
        X = df.copy()
        y = pd.DataFrame(index=df.index)
    return X, y

def recombine_features_and_target(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    y = y.reindex(X.index)  # Ensure matching index before concat
    return pd.concat([X, y], axis=1)

def initialize_session_state():
    if 'raw1_df' not in st.session_state:
        st.session_state.raw1_df = pd.DataFrame()  # Initialize as empty DataFrame
    if 'preprocessed_df' not in st.session_state:
        st.session_state.preprocessed_df = pd.DataFrame()
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None
    if 'preprocessing_steps_completed' not in st.session_state:
        st.session_state.preprocessing_steps_completed = {
            'missing_values': False,
            'outliers': False,
            'log_transform': False,
            'scaling': False,
            'encoding': False,
            'multicollinearity': False,
            'feature_engineering': False
        }
    if 'data_snapshots' not in st.session_state:
        st.session_state.data_snapshots = {
            'initial': None,
            'before_missing_handling': None,
            'before_outlier_handling': None,
            'before_log_transform': None,
            'before_scaling': None,
            'before_encoding': None,
            'before_multicollinearity': None,
            'before_feature_engineering_extraction': None,
            'before_feature_engineering_creation': None,
            'before_feature_engineering_selection': None,
        }
    if 'fitted_preprocessing_components' not in st.session_state:
        st.session_state.fitted_preprocessing_components = {
            'missing_value_handler': None,
            'outlier_detector': None,
            'log_transformer': None,
            'feature_scaler': None,
            'categorical_encoder': None,
            'multicollinearity_reducer': None,
            'pca_extractor': None,
            'feature_selector': None
        }
    # Initialize pipeline
    if 'fitted_preprocessing_pipeline' not in st.session_state:
        st.session_state.fitted_preprocessing_pipeline = None
        
    # Add these two lines to initialize outlier-related state variables
    if 'outlier_table' not in st.session_state:
        st.session_state.outlier_table = None
    if 'outlier_detector' not in st.session_state:
        st.session_state.outlier_detector = None
    
    # Rest of your initialization code...
    
    # Dummy data for demonstration if no data uploaded
    if st.session_state.raw1_df.empty:
        data = {
            'feature_A': np.random.rand(100) * 100,
            'feature_B': np.random.randint(1, 10, 100),
            'feature_C': np.random.choice(['X', 'Y', 'Z'], 100),
            'feature_D': np.random.randn(100),
            'categorical_col_1': np.random.choice(['Red', 'Green', 'Blue'], 100),
            'categorical_col_2': np.random.choice(['Small', 'Medium', 'Large'], 100),
            'target': np.random.choice([0, 1], 100)
        }
        # Introduce some missing values
        for _ in range(15):  # More missing values
            row = np.random.randint(0, 99)
            col = np.random.choice(list(data.keys()))
            data[col][row] = np.nan
        # Introduce some columns with >30% missing values
        data['feature_E_high_missing'] = np.full(100, np.nan)
        for _ in range(80):  # Make about 80% missing
            row = np.random.randint(0, 99)
            data['feature_E_high_missing'][row] = np.random.rand()
        
        # Introduce some outliers
        data['feature_A'][0] = 1000  # outlier
        data['feature_D'][1] = 50  # outlier
        data['feature_A'][2] = -500  # negative outlier

        st.session_state.raw1_df = pd.DataFrame(data)
        if st.session_state.preprocessed_df.empty:
            st.session_state.preprocessed_df = st.session_state.raw1_df.copy()
            st.session_state.data_snapshots['initial'] = st.session_state.raw1_df.copy()
            st.session_state.selected_target = 'target'  # Set default target for dummy data

# Path for saving processed data and pipeline components
PROCESSED_DATA_DIR = "./processed_data"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def generate_timestamped_filename(prefix, extension=".csv"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"

def download_dataframe(df, filename):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Data",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key="download_processed_data"
    )

def display_current_data_and_nan_details():
    st.write("**Current Data Preview:**")
    st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)

    missing_pct = st.session_state.preprocessed_df.isnull().mean() * 100
    all_nan_cols = st.session_state.preprocessed_df.columns[st.session_state.preprocessed_df.isnull().all()].tolist()
    missing_flag_series = st.session_state.preprocessed_df.isnull().any()

    missing_summary = pd.DataFrame({
        "Column": st.session_state.preprocessed_df.columns,
        "Missing Count": st.session_state.preprocessed_df.isnull().sum(),
        "% Missing": missing_pct.round(2),
        "Has Missing?": missing_flag_series.map({True: "✅", False: "❌"}),
        "Category": [
            "All NaNs (Non-Value)" if col in all_nan_cols else
            "<5%" if pct < 5 else "5–30%" if pct <= 30 else ">30%"
            for col, pct in zip(st.session_state.preprocessed_df.columns, missing_pct.values)
        ]
    })
    
    missing_summary_filtered = missing_summary[missing_summary["Missing Count"] > 0]

    if not missing_summary_filtered.empty:
        st.write("**Missing Value Summary (Current Dataset):**")
        st.dataframe(missing_summary_filtered, use_container_width=True)
        st.markdown("""
        <span style='color:green'><b><5% missing</b></span>: Simple imputation (mean/median/mode)<br>
        <span style='color:orange'><b>5–30% missing</b></span>: Advanced imputation (KNN/model-based)<br>
        <span style='color:red'><b>>30% missing</b></span>: Will be dropped unless critical, including 'Non-Value' features.<br>
        """, unsafe_allow_html=True)
    else:
        st.info("No missing values found in the current dataset.")

# Helper function for showing NaN rows (added for completeness)
def show_nan_rows(df_to_check, key_suffix=""):
    nan_rows_df = df_to_check[df_to_check.isnull().any(axis=1)]
    if not nan_rows_df.empty:
        st.write(f"**Rows with NaN Values {key_suffix.replace('_', ' ').capitalize()}:**")
        st.dataframe(nan_rows_df.head(), use_container_width=True)
        if len(nan_rows_df) > 5:
            st.info(f"Displaying first 5 of {len(nan_rows_df)} rows with missing values.")
    else:
        st.info(f"No rows with missing values detected {key_suffix.replace('_', ' ')}.")

# New intelligent_impute function to be used by the Streamlit app
def intelligent_impute(df_features_to_impute):
    mv_handler = MissingValueHandler()
    df_transformed = mv_handler.fit_transform(df_features_to_impute.copy())
    return df_transformed, mv_handler.imputation_summary, mv_handler.imputation_log

# --- Main App ---
def main():
    initialize_session_state()
    logger.info("Loading Preprocessing page")
    
    st.markdown("# 🛠️ Data Preprocessing Pipeline")
    st.write("Perform various preprocessing steps on your dataset.")

    # Check for uploaded data
    if 'raw1_df' not in st.session_state or st.session_state.raw1_df.empty:
        st.warning("Please upload a dataset on the 'Upload Dataset' page first. (Using dummy data for demonstration)")

    # Initialize preprocessed_df if needed (only if it's truly empty after all other logic)
    if st.session_state.preprocessed_df.empty and 'raw1_df' in st.session_state and not st.session_state.raw1_df.empty:
        st.session_state.preprocessed_df = st.session_state.raw1_df.copy()
        st.session_state.data_snapshots['initial'] = st.session_state.raw1_df.copy()

    # --- Overall Reset Button ---
    st.markdown("---")
    if st.button("🔄 **Reset All Preprocessing**", help="Resets the dataset to its initial uploaded state and clears all preprocessing steps."):
        st.session_state.preprocessed_df = st.session_state.raw1_df.copy()
        st.session_state.preprocessing_steps_completed = {
            'missing_values': False,
            'outliers': False,
            'log_transform': False,
            'scaling': False,
            'encoding': False,
            'multicollinearity': False,
            'feature_engineering': False
        }
        st.session_state.fitted_preprocessing_components = {
            'missing_value_handler': None,
            'outlier_detector': None,
            'log_transformer': None,
            'feature_scaler': None,
            'categorical_encoder': None,
            'multicollinearity_reducer': None,
            'pca_extractor': None,
            'feature_selector': None
        }
        st.session_state.fitted_preprocessing_pipeline = None
        st.session_state.data_snapshots = {
            'initial': st.session_state.raw1_df.copy(),  # Re-capture initial state
            'before_missing_handling': None,
            'before_outlier_handling': None,
            'before_log_transform': None,
            'before_scaling': None,
            'before_encoding': None,
            'before_multicollinearity': None,
            'before_feature_engineering_extraction': None,
            'before_feature_engineering_creation': None,
            'before_feature_engineering_selection': None,
        }
        st.rerun()  # Rerun to reflect the reset state


    # --- Target Variable Selection ---
    st.markdown("---")
    st.subheader("🎯 Select Target Variable")

    columns = st.session_state.preprocessed_df.columns.tolist()
    default_target_index = 0

    if AppSettings.TARGET_COLUMN in columns:
        default_target_index = columns.index(AppSettings.TARGET_COLUMN)
    elif st.session_state.get("selected_target") in columns:
        default_target_index = columns.index(st.session_state["selected_target"])

    st.session_state["selected_target"] = st.selectbox(
        "Choose the target variable for your analysis:",
        columns,
        index=default_target_index,
        key="target_selector"
    )

    # Set X and y
    if st.session_state["selected_target"]:
        st.info(f"Target variable set to: **{st.session_state['selected_target']}**")
        
        if st.session_state["selected_target"] in st.session_state.preprocessed_df.columns:
            st.session_state.X = st.session_state.preprocessed_df.drop(columns=[st.session_state["selected_target"]])
            st.session_state.y = st.session_state.preprocessed_df[st.session_state["selected_target"]]
        else:
            st.warning(f"Selected target column '{st.session_state['selected_target']}' not found in the DataFrame. Please select another.")
            st.session_state.X = st.session_state.preprocessed_df.copy()
            st.session_state.y = pd.Series([], dtype=object)
    else:
        st.warning("No target variable selected. Some preprocessing steps (like feature selection) might be limited.")
        st.session_state.X = st.session_state.preprocessed_df.copy()
        st.session_state.y = pd.Series([], dtype=object)
    # --- Save Button for Inference Features ---
    st.markdown("---")
    st.subheader("🧠 Inference Feature Saver")

    if st.button("💾 Save Inference Features"):
        FEATURE_SAVE_DIR = Path("/home/hari/Logistic_regression_main/src/models/infrence_features")
        FEATURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        try:
            df = st.session_state.X
            inference_features = df.columns.tolist()

            # Save plain feature list
            feature_filename = f"inference_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            feature_path = FEATURE_SAVE_DIR / feature_filename

            with open(feature_path, "w", encoding="utf-8") as f:
                json.dump(inference_features, f, indent=2)

            # --- Save schema with feature names and dtypes ---
            schema_filename = f"inference_schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            schema_path = FEATURE_SAVE_DIR / schema_filename

            schema = {
                "features": [
                    {"name": col, "dtype": str(df[col].dtype)}
                    for col in inference_features
                ]
            }

            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2)

            # Save download-ready version to session
            st.session_state['last_inference_feature_file'] = {
                "filename": feature_filename,
                "data": json.dumps(inference_features, indent=2)
            }
            st.session_state['last_inference_schema_file'] = {
                "filename": schema_filename,
                "data": json.dumps(schema, indent=2)
            }

            st.success(f"✅ Inference feature list saved to: `{feature_path}`")
            st.success(f"✅ Inference schema saved to: `{schema_path}`")

        except Exception as e:
            st.error(f"❌ Failed to save inference features/schema: {e}")

    # Show download buttons if files were saved
    if 'last_inference_feature_file' in st.session_state:
        st.download_button(
            label="⬇️ Download Inference Feature List (JSON)",
            data=st.session_state['last_inference_feature_file']['data'],
            file_name=st.session_state['last_inference_feature_file']['filename'],
            mime="application/json"
        )

    if 'last_inference_schema_file' in st.session_state:
        st.download_button(
            label="⬇️ Download Inference Feature Schema (JSON)",
            data=st.session_state['last_inference_schema_file']['data'],
            file_name=st.session_state['last_inference_schema_file']['filename'],
            mime="application/json"
        )

    # --- Show Current Data Overview ---
    display_current_data_and_nan_details()

    # --- Preprocessing Steps ---
    # 1️⃣ Handle Missing Values
    st.markdown("---")

    with st.expander(f"1️⃣ Handle Missing Values {'✅' if st.session_state.preprocessing_steps_completed.get('missing_values', False) else '❌'}", expanded=True):
        st.write("Detect and handle missing values in your dataset (deletion/imputation).")

        df = st.session_state['preprocessed_df']
        target_col = st.session_state.get('selected_target', None)
        X, y = separate_features_and_target(df, target_col)

        if 'show_missing_summary' not in st.session_state:
            st.session_state.show_missing_summary = False

        # 🚩 Flag rows with missing values
        if st.button("🚩 Flag Missing Values", key="btn_flag_missing"):
            st.session_state.show_missing_summary = True
            flagged_X = X.copy()
            flagged_X['Missing_Flag'] = X.isnull().any(axis=1).astype(int)

            st.session_state['preprocessed_df'] = recombine_features_and_target(flagged_X, y)
            st.success(f"✅ Flagged **{flagged_X['Missing_Flag'].sum()}** rows with missing values.")

            st.write("### ✅ Data After Flagging:")
            st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)

        # 🚀 Run Imputation
        if st.button("🚀 Run Intelligent Missing Value Handler", key="btn_missing"):
            st.session_state.data_snapshots['before_missing_handling'] = st.session_state.preprocessed_df.copy()
            try:
                before_nan_counts = X.isna().sum()

                # Drop 'Missing_Flag' if present to avoid its influence on imputation
                X_cleaned = X.drop(columns=['Missing_Flag'], errors='ignore')

                X_transformed, summary_table, imputation_log = intelligent_impute(X_cleaned)
                X_transformed.index = X.index  # Retain original index

                # Restore 'Missing_Flag' if it existed
                if 'Missing_Flag' in X.columns:
                    X_transformed['Missing_Flag'] = X['Missing_Flag']

                after_nan_counts = X_transformed.isna().sum()

                # Recombine and update state
                st.session_state.preprocessed_df = recombine_features_and_target(X_transformed, y)
                st.session_state.preprocessing_steps_completed['missing_values'] = True
                st.session_state.fitted_preprocessing_components['missing_value_handler'] = mv_handler

                st.success("✅ Missing values handled successfully!")

                # 📊 Imputation Report
                st.write("### Imputation Report")
                imputed_cols = summary_table[summary_table["Imputation Method"].str.contains("Imputation", na=False)]["Column"].tolist()
                dropped_cols = summary_table[summary_table["Imputation Method"].str.contains("dropped", na=False)]["Column"].tolist()

                if imputed_cols:
                    imputed_report = []
                    for col in imputed_cols:
                        imputed_report.append({
                            "Column": col,
                            "NaN Before": before_nan_counts.get(col, 0),
                            "NaN After": after_nan_counts.get(col, 0),
                            "Imputation Method": summary_table[summary_table["Column"] == col]["Imputation Method"].values[0]
                        })
                    st.write("**Imputed Columns:**")
                    st.dataframe(pd.DataFrame(imputed_report), use_container_width=True)
                else:
                    st.write("**No columns were imputed.**")

                if dropped_cols:
                    dropped_report = []
                    for col in dropped_cols:
                        dropped_report.append({
                            "Column": col,
                            "NaN Count Before Dropping": before_nan_counts.get(col, 0),
                            "Reason": summary_table[summary_table["Column"] == col]["Imputation Method"].values[0]
                        })
                    st.write("**Dropped Columns:**")
                    st.dataframe(pd.DataFrame(dropped_report), use_container_width=True)
                else:
                    st.write("**No columns were dropped.**")

                st.write("**Imputation Log:**")
                st.code('\n'.join(imputation_log))

                st.write("### ✅ Data After Missing Value Handling:")
                st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)

            except Exception as e:
                st.error(f"🚫 Missing value handling failed: {str(e)}")
                logger.error(f"Missing value handler failed: {str(e)}\n{traceback.format_exc()}")
        # Optional: Remove Missing_Flag column if it exists after imputation
        if 'Missing_Flag' in st.session_state.preprocessed_df.columns:
            st.markdown("### 🧹 Post-Imputation Cleanup")
            st.info("`Missing_Flag` column was useful for tracking but can be dropped before proceeding.")

            if st.button("🗑️ Drop 'Missing_Flag' Column", key="btn_drop_missing_flag"):
                st.session_state.preprocessed_df.drop(columns=['Missing_Flag'], inplace=True)
                st.success("✅ 'Missing_Flag' column removed successfully.")

                # Optional: Preview after drop
                st.write("### ✅ Data After Cleanup:")
                st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)


        # 📊 Preview
        if not st.session_state['preprocessed_df'].empty:
            st.write("---")
            st.write("### Current Data Preview (After last action in this section):")
            st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)
            show_nan_rows(st.session_state['preprocessed_df'], key_suffix="after_handling")
    # 2️⃣ Handle Outliers
    with st.expander(f"2️⃣ Handle Outliers {'✅' if st.session_state.preprocessing_steps_completed.get('outliers', False) else '❌'}", expanded=True):
        st.write("Detect and handle outliers using IQR or other methods. Preview, flag, or remove outliers from your dataset.")

        if st.session_state.preprocessed_df.empty:
            st.warning("Please load and preprocess your data in previous steps first.")
        else:
            df = st.session_state.preprocessed_df.copy()
            target_col = st.session_state.get('selected_target', None)

            outlier_factor = st.number_input(
                "IQR Factor (e.g., 1.5 for mild, 3 for extreme outliers)",
                min_value=0.1,
                value=AppSettings.OUTLIER_IQ_FACTOR,
                step=0.1,
                key="outlier_factor_input"
            )

            selected_detection_method_name = st.selectbox(
                "Select Outlier Detection Method:",
                options=[method.name for method in DetectionMethod],
                index=0,
                key="outlier_detection_method_select"
            )
            selected_detection_method = DetectionMethod[selected_detection_method_name]

            X, y = separate_features_and_target(df, target_col)
            numeric_X = X.select_dtypes(include=np.number)

            if target_col and target_col in numeric_X.columns:
                numeric_X = numeric_X.drop(columns=[target_col])

            # --- DETECT OUTLIERS ---
            if st.button("🔍 Detect Outliers", key="btn_outlier_detect"):
                try:
                    if numeric_X.empty:
                        st.warning("⚠️ No numeric columns found to detect outliers.")
                    else:
                        detector = OutlierDetector(method=selected_detection_method, factor=outlier_factor)
                        detector.fit(numeric_X)
                        summary = detector.detect_outliers_table(numeric_X)

                        st.session_state['outlier_detector'] = detector
                        st.session_state['outlier_table_summary'] = summary

                        st.subheader("📊 Outlier Detection Summary")
                        st.dataframe(summary, use_container_width=True)

                        flagged_rows = detector.get_outlier_rows(numeric_X)
                        st.success(f"✅ Detected {flagged_rows.shape[0]} rows with at least one outlier.")
                        st.info(f"Checked columns: {list(numeric_X.columns)}")

                except Exception as e:
                    st.error(f"🚫 Outlier detection failed: {e}")
                    logger.exception("Detection failed")

            # --- DISPLAY DETECTION REPORT ---
            if 'outlier_table_summary' in st.session_state and not st.session_state['outlier_table_summary'].empty:
                st.markdown("---")
                st.write("### Outlier Detection Report:")
                st.dataframe(st.session_state['outlier_table_summary'], use_container_width=True)

                if st.checkbox("Show sample of rows with outliers", key="show_outlier_rows_checkbox"):
                    if 'outlier_detector' in st.session_state:
                        sample_rows = st.session_state['outlier_detector'].get_outlier_rows(numeric_X)
                        if not sample_rows.empty:
                            st.dataframe(df.loc[sample_rows.index].head(10), use_container_width=True)
                            st.info(f"Showing first 10 of {sample_rows.shape[0]} rows with detected outliers.")
                        else:
                            st.warning("⚠️ No outlier rows found.")
                    else:
                        st.warning("Run 'Detect Outliers' first.")

            # --- FLAG OUTLIERS ---
            if st.button("🚩 Flag Outliers", key="btn_outlier_flag"):
                try:
                    if numeric_X.empty:
                        st.warning("⚠️ No numeric columns available to flag.")
                    else:
                        detector = OutlierDetector(
                            method=selected_detection_method,
                            factor=outlier_factor,
                            handling=OutlierHandlingMethod.FLAG
                        )
                        detector.fit(numeric_X)
                        flagged_df = detector.transform(X.copy())
                        st.session_state.fitted_preprocessing_components['outlier_handler'] = detector
                        if 'fitted_preprocessing_components' not in st.session_state:
                            st.session_state.fitted_preprocessing_components = {}
                        st.session_state.fitted_preprocessing_components['outlier_handler'] = detector

                        st.session_state.preprocessed_df = recombine_features_and_target(flagged_df, y)
                        st.session_state.preprocessing_steps_completed['outliers'] = True

                        total_flagged = flagged_df.filter(like='Outlier_Flag_').any(axis=1).sum()
                        st.success(f"✅ Outliers flagged! {total_flagged} rows have one or more flags.")

                        flag_cols = [col for col in flagged_df.columns if col.startswith('Outlier_Flag_')]
                        if flag_cols:
                            st.write("### Flag Summary")
                            flag_counts = flagged_df[flag_cols].sum().reset_index()
                            flag_counts.columns = ['Feature', 'Count']
                            flag_counts['Feature'] = flag_counts['Feature'].str.replace('Outlier_Flag_', '')
                            st.dataframe(flag_counts, use_container_width=True)

                            st.write("### Sample of Flagged Rows")
                            flagged_sample = flagged_df[flagged_df['Outlier_Flag_Any'] == 1]
                            if not flagged_sample.empty:
                                display_cols = [col for col in X.columns if col in flagged_sample.columns] + flag_cols + ['Outlier_Flag_Any']
                                st.dataframe(flagged_sample[display_cols].head(10), use_container_width=True)
                            else:
                                st.info("No flagged rows to show.")
                except Exception as e:
                    st.error(f"🚫 Flagging failed: {e}")
                    logger.exception("Flagging error")

            # --- REMOVE OUTLIERS ---
            if st.button("❌ Remove Outliers", key="btn_outlier_remove"):
                try:
                    original_rows = X.shape[0]
                    detector = OutlierDetector(
                        method=selected_detection_method,
                        factor=outlier_factor,
                        handling=OutlierHandlingMethod.REMOVE
                    )
                    detector.fit(numeric_X)
                    cleaned_X = detector.transform(X.copy())
                    cleaned_y = y.loc[cleaned_X.index]
                    st.session_state.fitted_preprocessing_components['outlier_handler'] = detector
                    if 'fitted_preprocessing_components' not in st.session_state:
                        st.session_state.fitted_preprocessing_components = {}
                    st.session_state.fitted_preprocessing_components['outlier_handler'] = detector



                    st.session_state.preprocessed_df = recombine_features_and_target(cleaned_X, cleaned_y)
                    st.session_state.preprocessing_steps_completed['outliers'] = True

                    rows_removed = original_rows - cleaned_X.shape[0]
                    st.success(f"✅ Outliers removed. {rows_removed} rows dropped.")
                except Exception as e:
                    st.error(f"🚫 Removal failed: {e}")
                    logger.exception("Removal error")
            # --- CLEANUP: DROP OUTLIER FLAGS MANUALLY ---
            existing_cols = st.session_state.preprocessed_df.columns
            flag_cols = [col for col in existing_cols if col.startswith('Outlier_Flag')]

            if flag_cols:
                st.markdown("### 🧹 Manual Cleanup")
                st.info(f"Outlier flag columns detected: {', '.join(flag_cols)}. You can remove them manually if needed.")

                if st.button("🗑️ Drop Outlier Flag Columns", key="btn_drop_outlier_flags"):
                    st.session_state.preprocessed_df.drop(columns=flag_cols, inplace=True)
                    st.success("✅ Outlier flag columns removed successfully.")

                    st.write("### ✅ Data After Manual Cleanup:")
                    st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)
                    st.caption(f"Data shape: {st.session_state.preprocessed_df.shape}")

            # --- Final Preview ---
            if not st.session_state.preprocessed_df.empty:
                st.markdown("---")
                st.subheader("📦 Final Data After Outlier Handling")
                st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)
                st.caption(f"Data shape: {st.session_state.preprocessed_df.shape}")

                    
    # 3️⃣ Distribution Corrections: Skewness & Kurtosis
    with st.expander(f"3️⃣ Distribution Corrections: Skewness & Kurtosis {'✅' if st.session_state.preprocessing_steps_completed.get('log_transform', False) else '❌'}", expanded=False):

        st.write("""
        Apply transformation to reduce skewness or improve distribution characteristics:
        - **Log**, **Square Root**, **Cube Root**, **Inverse**
        - **Box-Cox**, **Yeo-Johnson**, **Power**, **Winsorizing**, **Clipping**, **Constant Addition**
        - **Data Binning**, **Quantile Transform**, **Z-Score**
        """)

        col1, col2 = st.columns(2)

        with col1:
            log_method = st.selectbox(
                "Select Transformation Method",
                options=[
                    "log", "square-root", "cube-root", "inverse", "box-cox", "yeo-johnson",
                    "power", "winsorizing", "clipping", "constant-addition",
                    "data-binning", "quantile-transform", "z-score"
                ],
                index=0,
                help="Choose the mathematical transformation method to apply"
            )

            offset_value = st.number_input(
                "Offset Value (for handling zeros)",
                min_value=0.0,
                value=1.0,
                step=0.1,
                help="Value added to avoid log(0) or box-cox failure"
            )

        with col2:
            handle_zeros = st.selectbox(
                "How to handle zeros/negatives?",
                options=["offset", "ignore", "drop"],
                index=0,
                help="'offset' adds a constant, 'ignore' may cause errors, 'drop' removes problematic rows"
            )

            show_details = st.checkbox(
                "Show transformation details",
                value=False,
                help="Show additional information about the transformation"
            )

        numeric_cols = st.session_state.preprocessed_df.select_dtypes(include=np.number).columns.tolist()
        if st.session_state.selected_target in numeric_cols:
            numeric_cols.remove(st.session_state.selected_target)

        if not numeric_cols:
            st.warning("No numeric columns available for transformation.")
            features_to_transform = []
        else:
            features_to_transform = st.multiselect(
                "Select features to transform",
                options=numeric_cols,
                default=numeric_cols
            )

        # --- Skewness Analysis ---
        if st.button("📊 Analyze Feature Skewness", key="check_skewness"):
            df = st.session_state.preprocessed_df.copy()
            if not features_to_transform:
                st.warning("No features selected for skewness analysis.")
            else:
                try:
                    df_skew = df[features_to_transform].select_dtypes(include=np.number)
                    skew_vals = df_skew.skew().sort_values(ascending=False)
                    skew_df = pd.DataFrame({
                        "Feature": skew_vals.index,
                        "Skewness": skew_vals.values,
                        "Interpretation": [
                            "Highly skewed (transform recommended)" if abs(v) > 1 else
                            "Moderately skewed (transform may help)" if 0.5 < abs(v) <= 1 else
                            "Approx. symmetric" for v in skew_vals.values
                        ],
                        "Suggested Action": [
                            f"Apply {log_method}" if abs(v) > 1 else
                            f"Consider {log_method}" if 0.5 < abs(v) <= 1 else
                            "No action needed" for v in skew_vals.values
                        ]
                    })
                    st.subheader("🔍 Skewness Analysis Report")
                    st.dataframe(
                        skew_df.style.background_gradient(subset=["Skewness"], cmap='RdYlGn_r').format({"Skewness": "{:.2f}"})
                    )

                    # Skewness bar chart
                    st.subheader("📈 Skewness Distribution")
                    fig, ax = plt.subplots(figsize=(10, max(4, len(skew_vals) * 0.5)))
                    sns.barplot(x=skew_vals.index, y=skew_vals.values, ax=ax, palette="coolwarm")
                    ax.axhline(0, color='black', linestyle='--')
                    ax.axhline(1, color='red', linestyle=':')
                    ax.axhline(-1, color='red', linestyle=':')
                    ax.set_title("Feature Skewness")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    st.session_state.skewness_results = skew_df

                except Exception as e:
                    st.error(f"Skewness analysis failed: {str(e)}")
                    logger.error(traceback.format_exc())

        # --- Apply Transformation ---
        if st.button("🚀 Apply Transformation", key="apply_log_transform"):
            try:
                st.session_state.data_snapshots['before_log_transform'] = st.session_state.preprocessed_df.copy()
                st.session_state.transformed_features_list = features_to_transform.copy()

                df = st.session_state.preprocessed_df.copy()
                target_col = st.session_state.get('selected_target', None)

                transformer = LogTransformer(
                    method=log_method,
                    handle_zeros=handle_zeros,
                    offset=offset_value,
                    columns=features_to_transform
                )

                transformed_X = transformer.fit_transform(df[features_to_transform])
                remaining_X = df.drop(columns=features_to_transform)
                combined_df = pd.concat([transformed_X, remaining_X], axis=1)[df.columns]

                st.session_state.preprocessed_df = combined_df
                st.session_state.preprocessing_steps_completed['log_transform'] = True
                st.session_state.fitted_preprocessing_components['log_transformer'] = transformer

                st.success(f"✅ Applied {log_method} transformation to {len(features_to_transform)} feature(s).")

            except Exception as e:
                st.error(f"❌ Transformation failed: {str(e)}")
                logger.error(traceback.format_exc())
                if 'before_log_transform' in st.session_state.data_snapshots:
                    st.session_state.preprocessed_df = st.session_state.data_snapshots['before_log_transform']
                    st.error("🔄 Reverted to pre-transformation state.")

        # --- Comparison Visualization ---
        if (st.session_state.preprocessing_steps_completed.get('log_transform', False) and
            'transformed_features_list' in st.session_state and
            'before_log_transform' in st.session_state.data_snapshots):

            available_features = [f for f in st.session_state.transformed_features_list
                                if f in st.session_state.preprocessed_df.columns]

            if available_features:
                selected_feature = st.selectbox("Select Feature to Compare", options=available_features)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Before Transformation**")
                    fig, ax = plt.subplots()
                    sns.histplot(st.session_state.data_snapshots['before_log_transform'][selected_feature], kde=True, ax=ax)
                    ax.set_title("Before")
                    st.pyplot(fig)
                    skew_before = st.session_state.data_snapshots['before_log_transform'][selected_feature].skew()
                    st.caption(f"Skewness before: {skew_before:.2f}")

                with col2:
                    st.write("**After Transformation**")
                    fig, ax = plt.subplots()
                    sns.histplot(st.session_state.preprocessed_df[selected_feature], kde=True, ax=ax)
                    ax.set_title("After")
                    st.pyplot(fig)
                    skew_after = st.session_state.preprocessed_df[selected_feature].skew()
                    st.caption(f"Skewness after: {skew_after:.2f}")

                improvement = (abs(skew_before) - abs(skew_after)) / abs(skew_before) * 100 if skew_before else 0
                st.success(f"🎯 Skewness improved by {improvement:.1f}%")
            else:
                st.info("No valid transformed features available for visualization.")

    # 4️⃣ Feature Scaling
    with st.expander(f"4️⃣ Apply Feature Scaling {'✅' if st.session_state.preprocessing_steps_completed.get('scaling', False) else '❌'}", expanded=False):

        st.markdown("""
        ### 🎯 Why Scale Features?

        Scaling helps models converge faster and improves performance, especially for algorithms sensitive to feature magnitudes (e.g., SVMs, KNNs, gradient-based methods).

        **Choose the right scaler based on data distribution:**
        - 🟦 **StandardScaler (Z-score)**: Use when features follow **normal distribution** and have **varying units** **(Data Standardization)**.
        - 🟩 **MinMaxScaler**: Ideal when data doesn't follow normal distribution or contains **outliers** **(Data Normalization)**.
        """)

        scaler_type = st.radio(
            "Select Scaler Type:",
            ('StandardScaler', 'MinMaxScaler'),
            key="scaler_type_radio"
        )

        if st.button("🚀 Apply Feature Scaling", key="apply_scaling"):
            st.session_state.data_snapshots['before_scaling'] = st.session_state.preprocessed_df.copy()

            with st.spinner(f"Applying {scaler_type} scaling..."):
                try:
                    df = st.session_state.preprocessed_df.copy()
                    target_col = st.session_state.get('selected_target', None)
                    X, y = separate_features_and_target(df, target_col)

                    # Apply scaling only to numeric columns
                    X_numeric = X.select_dtypes(include='number')

                    if X_numeric.empty:
                        st.warning("⚠️ No numeric features found for scaling.")
                    else:
                        scaler_obj = FeatureScaler(method='standard' if scaler_type == 'StandardScaler' else 'minmax')
                        X_scaled = scaler_obj.fit_transform(X_numeric)
                        X_scaled.index = X.index

                        # Reattach non-numeric columns (if any)
                        non_numeric_X = X.select_dtypes(exclude='number')
                        X_final = pd.concat([X_scaled, non_numeric_X], axis=1)

                        st.session_state.preprocessed_df = recombine_features_and_target(X_final, y)
                        st.session_state.fitted_preprocessing_components['feature_scaler'] = scaler_obj
                        st.session_state.preprocessing_steps_completed['scaling'] = True

                        st.success("✅ Feature scaling completed!")
                        logger.info("Feature scaling applied successfully.")

                except Exception as e:
                    st.error(f"❌ Scaling failed: {str(e)}")
                    logger.error(f"Feature scaling error: {str(e)}\n{traceback.format_exc()}")

        st.markdown("### 📊 Preview of Scaled Data")
        st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)

    # 5️⃣ Categorical Encoding
    with st.expander(f"5️⃣ Encode Categorical Features {'✅' if st.session_state.preprocessing_steps_completed.get('encoding', False) else '❌'}"):
        st.write("Automatically detect categorical types (binary, nominal, ordinal, high-cardinality) and apply selected encoding strategy.")

        df = st.session_state.preprocessed_df.copy()
        target_col = st.session_state.get('selected_target', None)
        X, y = separate_features_and_target(df, target_col)

        current_categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if current_categorical_cols:
            # Step 1: Feature summary
            st.subheader("Categorical Feature Analysis")
            feature_summary = {}
            for col in current_categorical_cols:
                unique_vals = X[col].dropna().unique()
                feature_summary[col] = {
                    "#Unique": len(unique_vals),
                    "Values": unique_vals.tolist()[:5] + (["..."] if len(unique_vals) > 5 else [])
                }
            st.dataframe(pd.DataFrame(feature_summary).T)

            # Step 2: Optional ordinal mapping
            st.subheader("Define Ordinal Mappings (Optional)")
            selected_ordinal_col = st.selectbox(
                "Select a categorical column to define ordinal mapping:",
                ["None"] + current_categorical_cols,
                key="select_ordinal_col"
            )

            if selected_ordinal_col != "None":
                st.write(f"Define mapping for **'{selected_ordinal_col}'** (e.g., Small:1, Medium:2, Large:3)")
                default_map_str = ", ".join([f"{k}:{v}" for k, v in AppSettings.ORDINAL_FEATURES_MAP.get(selected_ordinal_col, {}).items()])
                mapping_input = st.text_area(
                    "Enter mapping as `category1:value1, category2:value2` (comma-separated):",
                    value=default_map_str,
                    key=f"ordinal_map_input_{selected_ordinal_col}"
                )

                parsed_map = {}
                try:
                    if mapping_input:
                        for item in mapping_input.split(','):
                            key, val = item.split(':')
                            parsed_map[key.strip()] = float(val.strip())
                    if parsed_map:
                        AppSettings.ORDINAL_FEATURES_MAP[selected_ordinal_col] = parsed_map
                        st.info(f"Mapping set for {selected_ordinal_col}: {parsed_map}")
                    else:
                        st.warning("No valid mapping entered. Default encoding will apply.")
                except Exception as e:
                    st.error(f"Invalid mapping format: {e}. Use category:value format.")

            # Step 3: Encoding strategy settings
            st.subheader("Encoding Strategy Settings")
            strategy = st.selectbox("High Cardinality Strategy:", ["frequency", "none"])
            handle_unknown = st.selectbox("Handle Unknown Categories:", ["ignore", "error"])
            max_cardinality = st.slider("Max Cardinality Threshold", 10, 100, 20)

            if st.button("🚀 Apply Categorical Encoding", key="apply_encoding"):
                st.session_state.data_snapshots['before_encoding'] = df.copy()

                with st.spinner("Applying categorical encoding..."):
                    try:
                        encoder = CategoricalEncoder(
                            ordinal_features_map=AppSettings.ORDINAL_FEATURES_MAP,
                            high_cardinality_strategy=strategy,
                            handle_unknown=handle_unknown,
                            max_cardinality=max_cardinality
                        )

                        X_encoded = encoder.fit_transform(X)
                        X_encoded.index = X.index

                        st.session_state.preprocessed_df = recombine_features_and_target(X_encoded, y)
                        st.session_state.fitted_preprocessing_components['categorical_encoder'] = encoder
                        st.session_state.preprocessing_steps_completed['encoding'] = True

                        st.success("✅ Categorical encoding completed!")
                        st.json({
                            "binary": encoder.binary_features,
                            "ordinal": encoder.ordinal_features,
                            "nominal": encoder.nominal_features,
                            "high_cardinality": encoder.high_cardinality_features
                        })
                    except Exception as e:
                        st.error(f"❌ Encoding failed: {str(e)}")
                        logger.error(f"Categorical encoding error: {str(e)}\n{traceback.format_exc()}")

        else:
            st.info("ℹ️ No categorical columns found in the dataset (excluding target variable).")

        st.markdown("### 📊 Preview of Encoded Data")
        st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)


    # 6️⃣ Multicollinearity Reduction
    with st.expander(f"6️⃣ Reduce Multicollinearity {'✅' if st.session_state.preprocessing_steps_completed.get('multicollinearity', False) else '❌'}", expanded=False):
        st.markdown("Identify and remove highly correlated numerical features to reduce multicollinearity. Choose between **VIF**-based filtering and optional **correlation-based** pruning.")

        df = st.session_state.preprocessed_df.copy()
        target_col = st.session_state.get('selected_target', None)
        X, y = separate_features_and_target(df, target_col)

        # ── Parameter controls ──
        col1, col2 = st.columns(2)
        vif_threshold = col1.slider("VIF Threshold", 1.0, 10.0, value=5.0, step=0.1)
        use_correlation = col2.checkbox("Use Correlation Elimination", value=False)

        if use_correlation:
            corr_threshold = st.slider("Correlation Threshold (absolute)", 0.5, 1.0, value=0.8, step=0.01)
        else:
            corr_threshold = 0.8

        # 🚀 Apply button
        if st.button("🚀 Apply Multicollinearity Reduction", key="apply_multicollinearity"):
            st.session_state.data_snapshots['before_multicollinearity'] = df.copy()

            with st.spinner("Reducing multicollinearity..."):
                try:
                    mc_reducer = MulticollinearityReducer(
                        vif_threshold=vif_threshold,
                        use_correlation=use_correlation,
                        corr_threshold=corr_threshold,
                        handle_missing='drop',
                        verbose=False
                    )

                    X_reduced = mc_reducer.fit_transform(X)
                    X_reduced.index = X.index

                    st.session_state.preprocessed_df = recombine_features_and_target(X_reduced, y)
                    st.session_state.fitted_preprocessing_components['multicollinearity_reducer'] = mc_reducer
                    st.session_state.preprocessing_steps_completed['multicollinearity'] = True

                    st.success("✅ Multicollinearity reduction completed!")

                    eliminated = mc_reducer.get_eliminated_features()
                    selected = mc_reducer.get_selected_features()

                    if eliminated:
                        st.warning(f"🧹 Eliminated features ({len(eliminated)}):")
                        st.code(", ".join(eliminated), language='text')
                    else:
                        st.info("✅ No features were eliminated at the selected thresholds.")

                    if selected:
                        st.success(f"🎯 Selected features retained ({len(selected)}):")
                        st.code(", ".join(selected), language='text')

                    with st.expander("🧾 Detailed Reduction Summary"):
                        st.code(mc_reducer.summary(), language='text')

                except Exception as e:
                    st.error(f"❌ Failed to reduce multicollinearity: {str(e)}")
                    logger.error(f"Multicollinearity reduction failed: {str(e)}\n{traceback.format_exc()}")

        # Final preview
        st.markdown("### 📊 Preview of Reduced Data")
        st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)
# --- Feature Engineering Section ---
    with st.expander(f"7️⃣ Feature Engineering {'✅' if st.session_state.preprocessing_steps_completed.get('feature_engineering', False) else '❌'}"):
        st.write("Perform advanced feature engineering steps like PCA, custom feature creation, and selection to optimize your dataset for modeling.")

        df = st.session_state.preprocessed_df.copy()
        target_col = st.session_state.get('selected_target')
        X, y = separate_features_and_target(df, target_col)

        # --- 7a. Feature Extraction (PCA) ---
        st.subheader("7a. Feature Extraction (PCA)")
        st.info("Reduce the dimensionality of your numeric data using Principal Component Analysis (PCA).")

        pca_n_components = st.slider(
            "Number of PCA Components (0 for automatic, -1 for all, 1+ for specific):",
            min_value=-1,
            max_value=min(X.select_dtypes(include=np.number).shape[1], 10),
            value=2,
            step=1,
            key="pca_n_components_slider",
            help="Set to 0 for automatic selection (explained variance), -1 to keep all components, or specify a number."
        )

        apply_pca = st.button("🚀 Apply PCA", key="apply_pca_button")

        # --- 7b. Custom Feature Creation ---
        st.subheader("7b. Feature Creation")
        st.info("Generate new features from existing numeric columns (e.g., sum, mean, product).")
        apply_feature_creation = st.button("🚀 Create Custom Features", key="apply_feature_creation_button")

        # --- 7c. Feature Selection ---
        st.subheader("7c. Feature Selection")
        st.info("Select the most relevant features to improve model performance and reduce noise.")

        selection_strategy = st.selectbox(
            "Selection Strategy",
            options=["univariate", "variance", "correlation"],
            index=0,
            key="feature_selection_strategy_select",
            help="Choose a strategy: 'univariate' (statistical tests), 'variance' (low variance removal), 'correlation' (correlation with target)."
        )

        selection_threshold = None
        if selection_strategy == "variance":
            selection_threshold = st.number_input(
                "Variance Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.3f",
                key="variance_threshold_input",
                help="Features with variance below this threshold will be removed."
            )
        elif selection_strategy == "correlation":
            selection_threshold = st.number_input(
                "Correlation Threshold (absolute value):",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                key="correlation_threshold_input",
                help="Features with absolute correlation to the target below this threshold will be removed."
            )

        apply_feature_selection = st.button("🚀 Apply Feature Selection", key="apply_feature_selection_button")

        # --- Action Logic ---
        if apply_pca:
            st.session_state.data_snapshots['before_feature_engineering_pca'] = df.copy()
            with st.spinner("Applying PCA..."):
                try:
                    numeric_X = X.select_dtypes(include=np.number)
                    if numeric_X.empty:
                        st.error("❌ No numeric columns available for PCA.")
                    else:
                        pca = FeatureExtractor(n_components=(None if pca_n_components in [0, -1] else pca_n_components))
                        X_pca = pca.fit_transform(numeric_X)
                        X_pca.index = X.index
                        non_numeric_X = X.select_dtypes(exclude=np.number)
                        X_combined = pd.concat([X_pca, non_numeric_X], axis=1)
                        st.session_state.preprocessed_df = recombine_features_and_target(X_combined, y)
                        st.session_state.fitted_preprocessing_components['pca_extractor'] = pca
                        st.session_state.preprocessing_steps_completed['feature_engineering'] = False
                        st.success(f"✅ PCA applied! {X_pca.shape[1]} components created.")
                except Exception as e:
                    st.error(f"❌ PCA failed: {str(e)}")
                    logger.error(traceback.format_exc())

        if apply_feature_creation:
            st.session_state.data_snapshots['before_feature_engineering_creation'] = df.copy()
            with st.spinner("Creating custom features..."):
                try:
                    creator = FeatureCreator()
                    X_created, new_features = creator.fit_transform(X)
                    X_created.index = X.index
                    st.session_state.preprocessed_df = recombine_features_and_target(X_created, y)
                    st.session_state.fitted_preprocessing_components['feature_creator'] = creator
                    st.session_state.preprocessing_steps_completed['feature_engineering'] = False
                    if new_features:
                        st.success(f"✅ Created features: {', '.join(new_features)}")
                    else:
                        st.info("ℹ️ No new features created. Possibly due to unsuitable input.")
                except Exception as e:
                    st.error(f"❌ Feature creation failed: {str(e)}")
                    logger.error(traceback.format_exc())

        if apply_feature_selection:
            st.session_state.data_snapshots['before_feature_engineering_selection'] = df.copy()
            with st.spinner("Applying feature selection..."):
                try:
                    X_numeric = X.select_dtypes(include=np.number)

                    if X_numeric.isnull().values.any():
                        imputer = SimpleImputer(strategy='mean')
                        X_numeric = pd.DataFrame(imputer.fit_transform(X_numeric), columns=X_numeric.columns, index=X_numeric.index)
                        if y is not None:
                            y = y.loc[X_numeric.index]

                    if X_numeric.empty:
                        st.error("❌ No numeric features available for selection.")
                    else:
                        selector = FeatureSelector(strategy=selection_strategy, threshold=selection_threshold or 0.0)
                        X_selected = selector.fit_transform(X_numeric, y if selection_strategy != "variance" else None)
                        X_selected.index = X.index
                        X_combined = pd.concat([X_selected, X.select_dtypes(exclude=np.number)], axis=1)
                        st.session_state.preprocessed_df = recombine_features_and_target(X_combined, y)
                        st.session_state.fitted_preprocessing_components['feature_selector'] = selector
                        st.session_state.preprocessing_steps_completed['feature_engineering'] = True
                        st.success(f"✅ Feature selection complete. Remaining features: {X_selected.shape[1]}")
                except Exception as e:
                    st.error(f"❌ Feature selection failed: {str(e)}")
                    logger.error(traceback.format_exc())

    # --- Generate timestamp once ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Download Preprocessed Dataset ---
    st.markdown("---")
    st.subheader("💾 Download Preprocessed Dataset")

    with st.expander("📤 Export Options", expanded=True):
        export_format = st.radio(
            "Select export format:",
            options=["CSV", "Excel", "JSON", "Parquet", "TXT"],
            horizontal=True
        )

        # Format-specific options
        if export_format == "CSV":
            index_opt = st.checkbox("Include index", value=False, key="csv_index")
            encoding = st.selectbox("Encoding", ["utf-8", "latin1", "utf-16"], index=0)
        elif export_format == "Excel":
            index_opt = st.checkbox("Include index", value=False, key="excel_index")
            sheet_name = st.text_input("Sheet name", value="Sheet1")
        elif export_format == "TXT":
            index_opt = st.checkbox("Include index", value=False, key="txt_index")
            encoding = st.selectbox("Encoding", ["utf-8", "latin1", "utf-16"], index=0)
            separator = st.text_input("Separator", value="\t")

        filename = st.text_input("Filename", value=f"preprocessed_data_{timestamp}.{export_format.lower()}")

        if st.button(f"⬇️ Download as {export_format}"):
            try:
                df = st.session_state.preprocessed_df
                if export_format == "CSV":
                    data = df.to_csv(index=index_opt).encode(encoding)
                    mime = "text/csv"
                elif export_format == "Excel":
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=index_opt)
                    buffer.seek(0)
                    data, mime = buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif export_format == "JSON":
                    data = df.to_json(orient="records").encode("utf-8")
                    mime = "application/json"
                elif export_format == "Parquet":
                    buffer = BytesIO()
                    df.to_parquet(buffer)
                    data, mime = buffer.getvalue(), "application/octet-stream"
                elif export_format == "TXT":
                    data = df.to_csv(index=index_opt, sep=separator).encode(encoding)
                    mime = "text/plain"

                st.download_button(
                    label=f"✅ Confirm Download {export_format}",
                    data=data,
                    file_name=filename,
                    mime=mime
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
                logger.error(traceback.format_exc())
    if st.button("💾 Save Preprocessing Pipeline", key="save_pipeline_button"):
        try:
            # 🕒 Generate timestamped filename
            filename_pipeline = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            preprocessing_steps = []

            # 1. Missing Values
            if (st.session_state.preprocessing_steps_completed.get('missing_values') and
                'missing_value_handler' in st.session_state.fitted_preprocessing_components):
                preprocessing_steps.append((
                    'missing_value_imputer',
                    st.session_state.fitted_preprocessing_components['missing_value_handler']
                ))

            # 2. Outliers
            if (st.session_state.preprocessing_steps_completed.get('outliers') and
                'outlier_handler' in st.session_state.fitted_preprocessing_components):
                preprocessing_steps.append((
                    'outlier_handler',
                    st.session_state.fitted_preprocessing_components['outlier_handler']
                ))

            # 3. Log Transform
            if (st.session_state.preprocessing_steps_completed.get('log_transform') and
                'log_transformer' in st.session_state.fitted_preprocessing_components):
                preprocessing_steps.append((
                    'log_transform',
                    st.session_state.fitted_preprocessing_components['log_transformer']
                ))

            # 4. Scaling
            if (st.session_state.preprocessing_steps_completed.get('scaling') and
                'feature_scaler' in st.session_state.fitted_preprocessing_components):
                preprocessing_steps.append((
                    'scaling',
                    st.session_state.fitted_preprocessing_components['feature_scaler']
                ))

            # 5. Encoding
            if (st.session_state.preprocessing_steps_completed.get('encoding') and
                'categorical_encoder' in st.session_state.fitted_preprocessing_components):
                preprocessing_steps.append((
                    'encoding',
                    st.session_state.fitted_preprocessing_components['categorical_encoder']
                ))

            # 6. Multicollinearity
            if (st.session_state.preprocessing_steps_completed.get('multicollinearity') and
                'multicollinearity_reducer' in st.session_state.fitted_preprocessing_components):
                preprocessing_steps.append((
                    'multicollinearity',
                    st.session_state.fitted_preprocessing_components['multicollinearity_reducer']
                ))

            # 7. Feature Engineering — Order: PCA → Feature Creation → Feature Selection
            if st.session_state.preprocessing_steps_completed.get("feature_engineering"):
                # ➤ PCA
                if "pca_extractor" in st.session_state.fitted_preprocessing_components:
                    pca = st.session_state.fitted_preprocessing_components["pca_extractor"]
                    preprocessing_steps.append(("pca", pca))

                # ➤ Feature Creation
                if "feature_creator" in st.session_state.fitted_preprocessing_components:
                    feature_creator = st.session_state.fitted_preprocessing_components["feature_creator"]
                    if not hasattr(feature_creator, 'original_features'):
                        feature_creator.original_features = [
                            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
                        ]
                    preprocessing_steps.append(("feature_creation", feature_creator))

                # ➤ Feature Selection
                if "feature_selector" in st.session_state.fitted_preprocessing_components:
                    feature_selector = st.session_state.fitted_preprocessing_components["feature_selector"]
                    preprocessing_steps.append(("feature_selector", feature_selector))

            # ✅ Create pipeline
            pipeline = SklearnPipeline(preprocessing_steps)

            # ➕ Set input schema (raw features)
            pipeline.expected_input_features = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]

            # ✅ Save pipeline
            pipeline_dir = Path("src/models/pipeline")
            pipeline_dir.mkdir(parents=True, exist_ok=True)
            pipeline_path = pipeline_dir / f"{filename_pipeline}.pkl"
            joblib.dump(pipeline, pipeline_path)
            st.session_state['fitted_preprocessing_pipeline'] = pipeline

            # ✅ Save processed schema
            schema_dir = Path("src/models/preprocessed_features")
            schema_dir.mkdir(parents=True, exist_ok=True)
            processed_schema_path = schema_dir / "schema_processed_features.json"

            try:
                # Try from pipeline output
                if hasattr(pipeline, "get_feature_names_out"):
                    processed_features = list(pipeline.get_feature_names_out())
                else:
                    processed_features = list(st.session_state['preprocessed_df'].columns)

                # ❌ Remove target column if present
                target_col = st.session_state.get('selected_target')
                if target_col in processed_features:
                    processed_features.remove(target_col)

            except Exception:
                target_col = st.session_state.get('selected_target')
                processed_features = [
                    col for col in st.session_state.get('preprocessed_df', pd.DataFrame()).columns
                    if col != target_col
                ]

            # Write to file
            with open(processed_schema_path, "w") as f:
                json.dump(processed_features, f)

            # ✅ Feedback
            st.success(f"✅ Pipeline saved to: {pipeline_path}")
            st.info(f"✅ Processed feature schema saved to: {processed_schema_path}")

        except Exception as e:
            st.error(f"❌ Pipeline save failed: {str(e)}")
            logger.error(traceback.format_exc())




    # --- Trigger EDA2 ---
    if st.button("📊 Visualize Processed Data (EDA2)"):
        try:
            if 'preprocessed_df' in st.session_state and not st.session_state.preprocessed_df.empty:
                st.session_state['raw2_df'] = st.session_state.preprocessed_df.copy()
                st.session_state['current_page'] = 'Visualize'
                st.switch_page("pages/4_Visualize.py")
            else:
                st.warning("No processed data available. Please preprocess first.")
        except Exception as e:
            st.error(f"Navigation failed: {e}")

# Entry point
if __name__ == "__main__":
    main()
