import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import traceback
import logging
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer  # For advanced imputation
from sklearn.decomposition import PCA 
from io import BytesIO
import json
from sklearn.impute import SimpleImputer


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
        st.session_state.fitted_preprocessing_components = {}
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
        st.session_state.fitted_preprocessing_components = {}
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
    elif st.session_state.selected_target in columns:
        default_target_index = columns.index(st.session_state.selected_target)
        
    st.session_state.selected_target = st.selectbox(
        "Choose the target variable for your analysis:",
        columns,
        index=default_target_index,
        key="target_selector"
    )

    if st.session_state.selected_target:
        st.info(f"Target variable set to: **{st.session_state.selected_target}**")
        if st.session_state.selected_target in st.session_state.preprocessed_df.columns:
            st.session_state.X = st.session_state.preprocessed_df.drop(columns=[st.session_state.selected_target])
            st.session_state.y = st.session_state.preprocessed_df[st.session_state.selected_target]
        else:
            st.warning(f"Selected target column '{st.session_state.selected_target}' not found in the DataFrame. Please select another.")
            st.session_state.X = st.session_state.preprocessed_df.copy()
            st.session_state.y = pd.Series([])  # Empty series
    else:
        st.warning("No target variable selected. Some preprocessing steps (like feature selection) might be limited.")
        st.session_state.X = st.session_state.preprocessed_df.copy()
        st.session_state.y = pd.Series([])  # Empty series

    display_current_data_and_nan_details()

    # --- Preprocessing Steps ---
    st.markdown("---")
    
    # 1️⃣ Handle Missing Values
    with st.expander(f"1️⃣ Handle Missing Values {'✅' if st.session_state.preprocessing_steps_completed['missing_values'] else '❌'}", expanded=True):
        st.write("Detect and handle missing values in your dataset (deletion/imputation).")

        df = st.session_state['preprocessed_df']
        target_col = st.session_state.get('selected_target', None)

        # Separate features and target
        if target_col and target_col in df.columns:
            features_df = df.drop(columns=[target_col], errors='ignore')
            target_df = df[[target_col]].copy()
        else:
            features_df = df.copy()
            target_df = pd.DataFrame(index=df.index)  # Empty target_df if no target or target not found

        if 'show_missing_summary' not in st.session_state:
            st.session_state.show_missing_summary = False

        # Flag rows with any missing value button
        if st.button("🚩 Flag Missing Values", key="btn_flag_missing"):
            st.session_state.show_missing_summary = True
            flagged_features_df = features_df.copy()
            flagged_features_df['Missing_Flag'] = features_df.isnull().any(axis=1).astype(int)

            if not target_df.empty:
                df_flagged = pd.concat([flagged_features_df, target_df], axis=1)
            else:
                df_flagged = flagged_features_df

            st.session_state['preprocessed_df'] = df_flagged

            st.success(f"✅ Flagged **{flagged_features_df['Missing_Flag'].sum()}** rows with missing values.")
            st.write("### ✅ Data After Flagging:")
            st.dataframe(df_flagged.head(), use_container_width=True)

        # Run intelligent imputation
        if st.button("🚀 Run Intelligent Missing Value Handler", key="btn_missing"):
            st.session_state.data_snapshots['before_missing_handling'] = st.session_state.preprocessed_df.copy()
            try:
                current_df = st.session_state['preprocessed_df']
                current_target_col = st.session_state.get('selected_target', None)

                if current_target_col and current_target_col in current_df.columns:
                    features_for_impute = current_df.drop(columns=[current_target_col], errors='ignore')
                    current_target_df = current_df[[current_target_col]].copy()
                else:
                    features_for_impute = current_df.copy()
                    current_target_df = pd.DataFrame(index=current_df.index)

                # Store NaN counts before imputation
                before_nan_counts = features_for_impute.isna().sum()

                # Ensure 'Missing_Flag' column is not used for imputation itself
                features_for_impute_cleaned = features_for_impute.drop(columns=['Missing_Flag'], errors='ignore')
                cleaned_features_df, summary_table, imputation_log = intelligent_impute(features_for_impute_cleaned)

                # Store NaN counts after imputation
                after_nan_counts = cleaned_features_df.isna().sum()

                # Re-add 'Missing_Flag' if it existed
                if 'Missing_Flag' in current_df.columns:
                    cleaned_features_df['Missing_Flag'] = current_df['Missing_Flag']

                imputed_cols = summary_table[summary_table["Imputation Method"].str.contains("Imputation", na=False)]["Column"].tolist()
                dropped_cols = summary_table[summary_table["Imputation Method"].str.contains("dropped", na=False)]["Column"].tolist()

                if not current_target_df.empty:
                    df_cleaned = pd.concat([cleaned_features_df, current_target_df], axis=1)
                else:
                    df_cleaned = cleaned_features_df

                st.session_state['preprocessed_df'] = df_cleaned
                st.session_state.preprocessing_steps_completed['missing_values'] = True

                st.success("✅ Missing values handled successfully!")

                # Display detailed imputation report
                st.write("### Imputation Report")

                # Create a DataFrame showing NaN counts before/after for imputed columns
                if imputed_cols:
                    imputed_report_data = []
                    for col in imputed_cols:
                        # Ensure the column exists in before_nan_counts and after_nan_counts
                        nan_before = before_nan_counts[col] if col in before_nan_counts else 0
                        nan_after = after_nan_counts[col] if col in after_nan_counts else 0
                        method = summary_table[summary_table["Column"] == col]["Imputation Method"].iloc[0] if not summary_table[summary_table["Column"] == col].empty else "N/A"
                        imputed_report_data.append({'Column': col, 'NaN Before': nan_before, 'NaN After': nan_after, 'Imputation Method': method})

                    imputed_report = pd.DataFrame(imputed_report_data)
                    st.write("**Imputed Columns:**")
                    st.dataframe(imputed_report, use_container_width=True)
                else:
                    st.write("**No columns were imputed**")

                if dropped_cols:
                    dropped_report_data = []
                    for col in dropped_cols:
                        nan_before = before_nan_counts[col] if col in before_nan_counts else 0
                        reason = summary_table[summary_table["Column"] == col]["Imputation Method"].iloc[0] if not summary_table[summary_table["Column"] == col].empty else "N/A"
                        dropped_report_data.append({'Column': col, 'NaN Count Before Dropping': nan_before, 'Reason': reason})
                    dropped_report = pd.DataFrame(dropped_report_data)
                    st.write("**Dropped Columns:**")
                    st.dataframe(dropped_report, use_container_width=True)
                else:
                    st.write("**No columns were dropped**")

                st.write("**Imputation Log:**")
                st.code('\n'.join(imputation_log))

                st.write("### ✅ Data After Missing Value Handling:")
                st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)

            except Exception as e:
                st.error(f"🚫 Missing value handling failed: {e}")

        # Preview cleaned data
        if not st.session_state['preprocessed_df'].empty:
            st.write("---")
            st.write("### Current Data Preview (After last action in this section):")
            st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)
            show_nan_rows(st.session_state['preprocessed_df'], key_suffix="after_handling")

    # 2️⃣ Handle Outliers
    with st.expander(f"2️⃣ Handle Outliers {'✅' if st.session_state.preprocessing_steps_completed.get('outliers', False) else '❌'}", expanded=True):
        st.write("Detect and handle outliers using IQR method. Preview, flag, or remove outliers from your dataset.")

        # Ensure preprocessed_df exists before proceeding
        if st.session_state.preprocessed_df.empty:
            st.warning("Please load and preprocess your data in previous steps first.")
        else:
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

            # 🟠 DETECT OUTLIERS
            if st.button("🔍 Detect Outliers", key="btn_outlier_detect"):
                try:
                    df = st.session_state['preprocessed_df'].copy()
                    target_col = st.session_state.get('selected_target', None)

                    features_df_numeric = df.select_dtypes(include=np.number)
                    if target_col and target_col in features_df_numeric.columns:
                        features_df_numeric = features_df_numeric.drop(columns=[target_col])

                    if features_df_numeric.empty:
                        st.warning("⚠️ No numeric columns found in the dataset to detect outliers.")
                    else:
                        detector = OutlierDetector(method=selected_detection_method, factor=outlier_factor)
                        detector.fit(features_df_numeric)
                        outlier_table_summary = detector.detect_outliers_table(features_df_numeric)

                        st.session_state['outlier_detector'] = detector
                        st.session_state['outlier_table_summary'] = outlier_table_summary

                        st.write("📊 **Outlier Detection Summary:**")
                        st.dataframe(outlier_table_summary, use_container_width=True)

                        temp_outlier_rows_df = detector.get_outlier_rows(features_df_numeric)
                        st.success(f"✅ Outlier detection completed! Found {temp_outlier_rows_df.shape[0]} unique rows with at least one outlier.")
                        st.info(f"Columns checked: {features_df_numeric.columns.tolist()}")
                except Exception as e:
                    st.error(f"🚫 Outlier detection failed: {e}")
                    if 'preprocessed_df' in st.session_state:
                        st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)

            if 'outlier_table_summary' in st.session_state and not st.session_state['outlier_table_summary'].empty:
                st.markdown("---")
                st.write("### Outlier Detection Report:")
                st.dataframe(st.session_state['outlier_table_summary'], use_container_width=True)

                if st.checkbox("Show sample of rows with outliers", key="show_outlier_rows_checkbox"):
                    if 'outlier_detector' in st.session_state:
                        df_current = st.session_state.preprocessed_df.copy()
                        target_col = st.session_state.get('selected_target', None)
                        features_df_numeric = df_current.select_dtypes(include=np.number)
                        if target_col and target_col in features_df_numeric.columns:
                            features_df_numeric = features_df_numeric.drop(columns=[target_col])

                        sample_outlier_rows = st.session_state['outlier_detector'].get_outlier_rows(features_df_numeric)
                        if not sample_outlier_rows.empty:
                            st.write("**Sample of Rows with Detected Outliers:**")
                            st.dataframe(df_current.loc[sample_outlier_rows.index].head(10), use_container_width=True)
                            st.info(f"Displaying first 10 of {len(sample_outlier_rows)} outlier rows.")
                        else:
                            st.warning("⚠️ No valid outlier rows found to display based on the last detection.")
                    else:
                        st.warning("Please run 'Detect Outliers' first to view sample rows.")

            # 🟠 FLAG OUTLIERS
            if st.button("🚩 Flag Outliers", key="btn_outlier_flag"):
                if st.session_state.preprocessed_df.empty:
                    st.warning("⚠️ Please load and preprocess your data first.")
                elif 'outlier_detector' not in st.session_state:
                    st.warning("⚠️ Please run 'Detect Outliers' first to configure the detector.")
                else:
                    try:
                        df_current = st.session_state.preprocessed_df.copy()
                        detector_for_flagging = OutlierDetector(
                            method=selected_detection_method,
                            factor=outlier_factor,
                            handling=OutlierHandlingMethod.FLAG
                        )

                        target_col = st.session_state.get('selected_target', None)
                        temp_features_df = df_current.select_dtypes(include=np.number)
                        if target_col and target_col in temp_features_df.columns:
                            temp_features_df = temp_features_df.drop(columns=[target_col])

                        if temp_features_df.empty:
                            st.warning("⚠️ No numeric columns available to flag outliers.")
                            return

                        detector_for_flagging.fit(temp_features_df)
                        df_flagged = detector_for_flagging.transform(df_current)

                        st.session_state.preprocessed_df = df_flagged
                        st.session_state.preprocessing_steps_completed['outliers'] = True

                        total_flagged_rows = df_flagged.filter(like='Outlier_Flag_').any(axis=1).sum()
                        st.success(f"✅ Outliers flagged successfully! {total_flagged_rows} rows have at least one outlier flag.")

                        st.write("### Flagging Details:")
                        flag_cols = [col for col in df_flagged.columns if col.startswith('Outlier_Flag_') and col != 'Outlier_Flag_Any']
                        if flag_cols:
                            st.write("**Outlier Flag Counts per Column:**")
                            flag_counts = df_flagged[flag_cols].sum().reset_index()
                            flag_counts.columns = ['Flag Column', 'Count True']
                            flag_counts['Flag Column'] = flag_counts['Flag Column'].str.replace('Outlier_Flag_', '')
                            st.dataframe(flag_counts, use_container_width=True)

                            st.write("**Sample of Flagged Rows (with original values and flags):**")
                            flagged_sample_df = df_flagged[df_flagged['Outlier_Flag_Any'] == 1]
                            if not flagged_sample_df.empty:
                                original_numeric_cols_with_flags = [c.replace('Outlier_Flag_', '') for c in flag_cols]
                                display_cols_for_sample = [col for col in original_numeric_cols_with_flags if col in df_current.columns] + flag_cols
                                if 'Outlier_Flag_Any' in df_flagged.columns:
                                    display_cols_for_sample.append('Outlier_Flag_Any')
                                st.dataframe(flagged_sample_df[display_cols_for_sample].head(10), use_container_width=True)
                                st.info(f"Displaying first 10 of {flagged_sample_df.shape[0]} rows with outliers flagged.")
                            else:
                                st.info("No rows were flagged as outliers with the current settings.")
                        else:
                            st.info("No numeric columns found or no outliers detected to flag.")
                    except Exception as e:
                        st.error(f"🚫 Flagging outliers failed: {e}")
                        logger.exception("Error during outlier flagging:")

            # 🟠 REMOVE OUTLIERS
            if st.button("❌ Remove Outliers", key="btn_outlier_remove"):
                if st.session_state.preprocessed_df.empty:
                    st.warning("⚠️ Please load and preprocess your data first.")
                elif 'outlier_detector' not in st.session_state:
                    st.warning("⚠️ Please run 'Detect Outliers' first to configure the detector.")
                else:
                    try:
                        df_current = st.session_state.preprocessed_df.copy()
                        original_rows = df_current.shape[0]

                        detector_for_removal = OutlierDetector(
                            method=selected_detection_method,
                            factor=outlier_factor,
                            handling=OutlierHandlingMethod.REMOVE
                        )

                        target_col = st.session_state.get('selected_target', None)
                        temp_features_df = df_current.select_dtypes(include=np.number)
                        if target_col and target_col in temp_features_df.columns:
                            temp_features_df = temp_features_df.drop(columns=[target_col])

                        if temp_features_df.empty:
                            st.warning("⚠️ No numeric columns available to remove outliers from.")
                            return

                        detector_for_removal.fit(temp_features_df)
                        df_cleaned = detector_for_removal.transform(df_current)

                        st.session_state.preprocessed_df = df_cleaned
                        st.session_state.preprocessing_steps_completed['outliers'] = True
                        rows_removed = original_rows - st.session_state.preprocessed_df.shape[0]
                        st.success(f"✅ Outliers removed successfully! ({rows_removed} rows removed)")

                    except Exception as e:
                        st.error(f"🚫 Outlier removal failed: {e}")
                        logger.exception("Error during outlier removal:")

            if not st.session_state['preprocessed_df'].empty:
                st.markdown("---")
                st.write("### Current Data Preview:")
                st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)
                st.write(f"DataFrame Shape: {st.session_state['preprocessed_df'].shape}")

    # 3️⃣ Log Transformation
    with st.expander(f"3️⃣ Apply Log Transformation {'✅' if st.session_state.preprocessing_steps_completed.get('log_transform', False) else '❌'}", expanded=False):
        st.write("""
        Apply transformation to reduce skewness in numerical features:
        - **Log**: Simple log transform (works only with positive values)
        - **Box-Cox**: Power transform (requires positive values)
        - **Yeo-Johnson**: Extended power transform (works with any values)
        """)

        st.caption("""
        Skewness measures asymmetry in data distribution.
        Values between -0.5 and 0.5 are considered symmetrical.
        Values beyond ±1 indicate high skewness that may benefit from transformation.
        """)

        col1, col2 = st.columns(2)

        with col1:
            log_method = st.selectbox(
                "Select Transformation Method",
                options=["log", "box-cox", "yeo-johnson"],
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

        numeric_cols = []
        if not st.session_state.preprocessed_df.empty:
            numeric_cols = st.session_state.preprocessed_df.select_dtypes(include=np.number).columns.tolist()
            if st.session_state.selected_target and st.session_state.selected_target in numeric_cols:
                numeric_cols.remove(st.session_state.selected_target)

        if not numeric_cols:
            st.warning("No numeric columns available for transformation. Please load data with numeric features.")
            features_to_transform = []
        else:
            features_to_transform = st.multiselect(
                "Select features to transform (default: all numeric)",
                options=numeric_cols,
                default=numeric_cols
            )

        if st.button("📊 Analyze Feature Skewness", key="check_skewness"):
            if 'preprocessed_df' not in st.session_state or st.session_state.preprocessed_df.empty:
                st.error("No data available for analysis. Please load data first.")
            elif not features_to_transform:
                st.warning("No features selected for skewness analysis.")
            else:
                df_for_skewness = st.session_state.preprocessed_df[features_to_transform].select_dtypes(include=np.number)
                if df_for_skewness.empty:
                    st.warning("No numeric columns found in selected features to compute skewness.")
                else:
                    try:
                        with st.spinner("Calculating skewness..."):
                            skew_vals = df_for_skewness.skew().sort_values(ascending=False)
                            skew_df = pd.DataFrame({
                                "Feature": skew_vals.index,
                                "Skewness": skew_vals.values,
                                "Interpretation": [
                                    "Highly skewed (transform recommended)" if abs(val) > 1.0 else
                                    "Moderately skewed (transform may help)" if 0.5 < abs(val) <= 1.0 else
                                    "Approx. symmetric (no transform needed)"
                                    for val in skew_vals.values
                                ],
                                "Suggested Action": [
                                    f"Apply {log_method}" if abs(val) > 1.0 else
                                    f"Consider {log_method}" if 0.5 < abs(val) <= 1.0 else
                                    "No action needed"
                                    for val in skew_vals.values
                                ]
                            })

                            st.subheader("🔍 Skewness Analysis Report")
                            st.dataframe(
                                skew_df.style.background_gradient(
                                    subset=['Skewness'],
                                    cmap='RdYlGn_r',
                                    vmin=-3,
                                    vmax=3
                                ).format({'Skewness': '{:.2f}'})
                            )

                            st.session_state.skewness_results = skew_df

                            st.subheader("📈 Skewness Distribution")
                            fig, ax = plt.subplots(figsize=(10, max(4, len(skew_vals) * 0.5)))
                            sns.barplot(x=skew_vals.index, y=skew_vals.values, ax=ax, palette="coolwarm")
                            plt.xticks(rotation=45, ha='right')
                            plt.axhline(0, color='black', linestyle='--')
                            plt.axhline(0.5, color='orange', linestyle=':')
                            plt.axhline(-0.5, color='orange', linestyle=':')
                            plt.axhline(1.0, color='red', linestyle=':')
                            plt.axhline(-1.0, color='red', linestyle=':')
                            plt.title("Feature Skewness")
                            plt.tight_layout()
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error calculating skewness: {str(e)}")
                        logger.error(f"Skewness calculation error: {str(e)}\n{traceback.format_exc()}")

        if st.button("🚀 Apply Transformation", key="apply_log_transform"):
            if 'preprocessed_df' not in st.session_state or st.session_state.preprocessed_df.empty:
                st.error("No data available for transformation. Please load data first.")
            elif not features_to_transform:
                st.warning("No features selected for transformation. Please select at least one feature.")
            else:
                try:
                    st.session_state.data_snapshots['before_log_transform'] = st.session_state.preprocessed_df.copy()
                    st.session_state.transformed_features_list = features_to_transform.copy()

                    with st.spinner(f"Applying {log_method} transformation..."):
                        df = st.session_state.preprocessed_df.copy()
                        target_col = st.session_state.get('selected_target', None)

                        non_transformed_cols = [col for col in df.columns
                                                if col not in features_to_transform and
                                                (not target_col or col != target_col)]

                        features_df = df[features_to_transform].copy()
                        other_cols_df = df[non_transformed_cols].copy() if non_transformed_cols else pd.DataFrame(index=df.index)
                        target_df = df[[target_col]] if target_col and target_col in df.columns else pd.DataFrame(index=df.index)

                        log_transformer = LogTransformer(
                            method=log_method,
                            handle_zeros=handle_zeros,
                            offset=offset_value,
                            columns=features_to_transform
                        )

                        transformed_features_df = log_transformer.fit_transform(features_df)

                        transformed_df = pd.concat([
                            transformed_features_df,
                            other_cols_df,
                            target_df
                        ], axis=1).reindex(df.index)

                        transformed_df = transformed_df[df.columns]

                        st.session_state.preprocessed_df = transformed_df
                        st.session_state.preprocessing_steps_completed['log_transform'] = True
                        st.session_state.transformed_features = features_to_transform

                        st.success(f"✅ Successfully applied {log_method} transformation to {len(features_to_transform)} feature(s): {', '.join(features_to_transform)}")

                except ValueError as e:
                    st.error(f"Transformation error: {str(e)}")
                    logger.error(f"Transformation error: {str(e)}\n{traceback.format_exc()}")
                    if 'before_log_transform' in st.session_state.data_snapshots:
                        st.session_state.preprocessed_df = st.session_state.data_snapshots['before_log_transform']
                        st.error("❌ Transformation failed. Reverted to pre-transformation state.")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    logger.error(f"Unexpected transformation error: {str(e)}\n{traceback.format_exc()}")
                    if 'before_log_transform' in st.session_state.data_snapshots:
                        st.session_state.preprocessed_df = st.session_state.data_snapshots['before_log_transform']
                        st.error("❌ Transformation failed. Reverted to pre-transformation state.")

        if (st.session_state.preprocessing_steps_completed.get('log_transform', False) and
            'transformed_features_list' in st.session_state and
            st.session_state.transformed_features_list and
            'before_log_transform' in st.session_state.data_snapshots and
            not st.session_state.data_snapshots['before_log_transform'].empty):

            available_features = [f for f in st.session_state.transformed_features_list
                                  if f in st.session_state.data_snapshots['before_log_transform'].columns and
                                  f in st.session_state.preprocessed_df.columns]

            if not available_features:
                st.warning("No transformed features available for comparison.")
            else:
                if ('selected_feature_for_comparison' not in st.session_state or
                        st.session_state.selected_feature_for_comparison not in available_features):
                    st.session_state.selected_feature_for_comparison = available_features[0]

                selected_feature = st.selectbox(
                    "Select feature to view transformation effect",
                    options=available_features,
                    index=available_features.index(st.session_state.selected_feature_for_comparison),
                    key='feature_comparison_select'
                )

                st.session_state.selected_feature_for_comparison = selected_feature

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before Transformation**")
                    fig_before, ax_before = plt.subplots(figsize=(6, 4))
                    sns.histplot(
                        st.session_state.data_snapshots['before_log_transform'][selected_feature],
                        ax=ax_before,
                        kde=True,
                        color='skyblue'
                    )
                    ax_before.set_title(f"Before: {selected_feature}")
                    ax_before.set_xlabel(selected_feature)
                    ax_before.set_ylabel("Frequency")
                    st.pyplot(fig_before)

                    skew_before = st.session_state.data_snapshots['before_log_transform'][selected_feature].skew()
                    st.caption(f"Skewness before: {skew_before:.2f}")

                with col2:
                    st.write("**After Transformation**")
                    fig_after, ax_after = plt.subplots(figsize=(6, 4))
                    sns.histplot(
                        st.session_state.preprocessed_df[selected_feature],
                        ax=ax_after,
                        kde=True,
                        color='salmon'
                    )
                    ax_after.set_title(f"After: {selected_feature}")
                    ax_after.set_xlabel(selected_feature)
                    ax_after.set_ylabel("Frequency")
                    st.pyplot(fig_after)

                    skew_after = st.session_state.preprocessed_df[selected_feature].skew()
                    st.caption(f"Skewness after: {skew_after:.2f}")

                improvement = (abs(skew_before) - abs(skew_after)) / abs(skew_before) * 100
                st.success(f"Skewness improved by {improvement:.1f}% (from {skew_before:.2f} to {skew_after:.2f})")

        elif st.session_state.preprocessing_steps_completed.get('log_transform', False) and not st.session_state.transformed_features_list:
            st.warning("No features were selected for transformation. Please apply transformation to view comparison.")
        elif st.session_state.preprocessing_steps_completed.get('log_transform', False) and 'before_log_transform' not in st.session_state.data_snapshots:
            st.warning("Original data snapshot missing. Please re-apply transformation to enable comparison.")

    # 4️⃣ Feature Scaling
    with st.expander(f"4️⃣ Apply Feature Scaling {'✅' if st.session_state.preprocessing_steps_completed['scaling'] else '❌'}"):
        st.write("Scale numerical features using StandardScaler (Z-score normalization) or MinMaxScaler.")
        
        scaler_type = st.radio(
            "Select Scaler Type:",
            ('StandardScaler', 'MinMaxScaler'),
            key="scaler_type_radio"
        )
        
        if st.button("🚀 Apply Feature Scaling", key="apply_scaling"):
            st.session_state.data_snapshots['before_scaling'] = st.session_state.preprocessed_df.copy()
            with st.spinner(f"Applying {scaler_type} scaling..."):
                # Separate features and target
                df = st.session_state.preprocessed_df.copy()
                target_col = st.session_state.get('selected_target', None)
                
                if target_col and target_col in df.columns:
                    features_df = df.drop(columns=[target_col])
                    target_df = df[[target_col]]
                else:
                    features_df = df.copy()
                    target_df = pd.DataFrame(index=df.index)
                
                # Apply scaling to features only
                scaler_obj = FeatureScaler(method='standard' if scaler_type == 'StandardScaler' else 'minmax')

                scaled_features_df = scaler_obj.fit_transform(features_df)
                
                # Recombine with target
                if not target_df.empty:
                    scaled_df = pd.concat([scaled_features_df, target_df], axis=1)
                else:
                    scaled_df = scaled_features_df
                
                st.session_state.preprocessed_df = scaled_df
                st.session_state.fitted_preprocessing_components['feature_scaler'] = scaler_obj
                st.session_state.preprocessing_steps_completed['scaling'] = True
                st.success("Feature scaling completed!")
                logger.info("Feature scaling completed.")
        
        st.write("**Current Data Preview after Scaling:**")
        st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)

    # 5️⃣ Categorical Encoding
    with st.expander(f"5️⃣ Encode Categorical Features {'✅' if st.session_state.preprocessing_steps_completed.get('encoding', False) else '❌'}"):
        st.write("Automatically detect categorical types (binary, nominal, ordinal, high-cardinality) and apply selected encoding strategy.")

        # Get categorical columns excluding target if it's categorical
        current_categorical_cols = st.session_state.preprocessed_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if st.session_state.selected_target in current_categorical_cols:
            current_categorical_cols.remove(st.session_state.selected_target)

        if current_categorical_cols:
            # Show feature summary
            st.subheader("Categorical Feature Analysis")
            feature_summary = {}
            for col in current_categorical_cols:
                unique_vals = st.session_state.preprocessed_df[col].dropna().unique()
                feature_summary[col] = {
                    "#Unique": len(unique_vals),
                    "Values": unique_vals.tolist()[:5] + (["..."] if len(unique_vals) > 5 else [])
                }
            st.dataframe(pd.DataFrame(feature_summary).T)

            # Define mapping UI
            st.subheader("Define Ordinal Mappings (Optional)")
            selected_ordinal_col = st.selectbox(
                "Select a categorical column to define ordinal mapping:",
                ["None"] + current_categorical_cols,
                key="select_ordinal_col"
            )

            if selected_ordinal_col != "None":
                st.write(f"Define mapping for **'{selected_ordinal_col}'** (e.g., Small:1, Medium:2, Large:3)")
                mapping_input = st.text_area(
                    "Enter mapping as `category1:value1, category2:value2` (comma-separated):",
                    value=", ".join([f"{k}:{v}" for k, v in AppSettings.ORDINAL_FEATURES_MAP.get(selected_ordinal_col, {}).items()]),
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

            # Select encoding strategies
            st.subheader("Encoding Strategy Settings")
            strategy = st.selectbox("High Cardinality Strategy:", ["frequency", "none"])
            handle_unknown = st.selectbox("Handle Unknown Categories:", ["ignore", "error"])
            max_cardinality = st.slider("Max Cardinality Threshold", 10, 100, 20)

            if st.button("🚀 Apply Categorical Encoding", key="apply_encoding"):
                st.session_state.data_snapshots['before_encoding'] = st.session_state.preprocessed_df.copy()

                with st.spinner("Applying categorical encoding..."):
                    # Separate features and target
                    df = st.session_state.preprocessed_df.copy()
                    target_col = st.session_state.get('selected_target', None)
                    
                    if target_col and target_col in df.columns:
                        features_df = df.drop(columns=[target_col])
                        target_df = df[[target_col]]
                    else:
                        features_df = df.copy()
                        target_df = pd.DataFrame(index=df.index)
                    
                    # Apply encoding to features only
                    encoder = CategoricalEncoder(
                        ordinal_features_map=AppSettings.ORDINAL_FEATURES_MAP,
                        high_cardinality_strategy=strategy,
                        handle_unknown=handle_unknown,
                        max_cardinality=max_cardinality
                    )
                    encoded_features_df = encoder.fit_transform(features_df)
                    
                    # Recombine with target
                    if not target_df.empty:
                        encoded_df = pd.concat([encoded_features_df, target_df], axis=1)
                    else:
                        encoded_df = encoded_features_df
                    
                    st.session_state.preprocessed_df = encoded_df
                    st.session_state.fitted_preprocessing_components['categorical_encoder'] = encoder
                    st.session_state.preprocessing_steps_completed['encoding'] = True

                    st.success("Categorical encoding completed!")
                    st.write("Feature types detected:")
                    st.json({
                        "binary": encoder.binary_features,
                        "ordinal": encoder.ordinal_features,
                        "nominal": encoder.nominal_features,
                        "high_cardinality": encoder.high_cardinality_features
                    })

        else:
            st.info("No categorical columns found in the dataset (excluding target variable).")

        if 'preprocessed_df' in st.session_state:
            st.write("**Data Preview after Encoding:**")
            st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)

    # 6️⃣ Multicollinearity Reduction
    with st.expander(f"6️⃣ Reduce Multicollinearity {'✅' if st.session_state.preprocessing_steps_completed['multicollinearity'] else '❌'}"):
        st.write("Identify and remove highly correlated numerical features to reduce multicollinearity.")

        # Parameter controls
        col1, col2 = st.columns(2)
        vif_threshold = col1.slider("VIF Threshold:", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        use_correlation = col2.checkbox("Use Correlation Elimination", value=False)

        if use_correlation:
            corr_threshold = st.slider("Correlation Threshold (absolute):", min_value=0.5, max_value=1.0, value=0.8, step=0.01)
        else:
            corr_threshold = 0.8  # fallback

        if st.button("🚀 Apply Multicollinearity Reduction", key="apply_multicollinearity"):
            st.session_state.data_snapshots['before_multicollinearity'] = st.session_state.preprocessed_df.copy()

            with st.spinner("Applying multicollinearity reduction..."):
                # Separate features and target
                df = st.session_state.preprocessed_df.copy()
                target_col = st.session_state.get('selected_target', None)
                
                if target_col and target_col in df.columns:
                    features_df = df.drop(columns=[target_col])
                    target_df = df[[target_col]]
                else:
                    features_df = df.copy()
                    target_df = pd.DataFrame(index=df.index)
                
                # Apply multicollinearity reduction to features only
                mc_reducer = MulticollinearityReducer(
                    vif_threshold=vif_threshold,
                    use_correlation=use_correlation,
                    corr_threshold=corr_threshold
                )

                reduced_features_df = mc_reducer.fit_transform(features_df)
                
                # Recombine with target
                if not target_df.empty:
                    reduced_df = pd.concat([reduced_features_df, target_df], axis=1)
                else:
                    reduced_df = reduced_features_df
                
                st.session_state.preprocessed_df = reduced_df
                st.session_state.fitted_preprocessing_components['multicollinearity_reducer'] = mc_reducer
                st.session_state.preprocessing_steps_completed['multicollinearity'] = True
                st.success("Multicollinearity reduction completed!")
                logger.info("Multicollinearity reduction completed.")

                if mc_reducer.get_eliminated_features():
                    st.info(f"Dropped features: {', '.join(mc_reducer.get_eliminated_features())}")
                else:
                    st.info("No features were dropped at the given thresholds.")

                st.code(mc_reducer.summary())

        # Display current preview
        st.write("**Current Data Preview after Multicollinearity Reduction:**")
        st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)

 # 7️⃣ Feature Engineering
    with st.expander(f"7️⃣ Feature Engineering {'✅' if st.session_state.preprocessing_steps_completed['feature_engineering'] else '❌'}"):
        st.write("Perform advanced feature engineering steps like PCA, custom feature creation, and selection.")

        # 7a. Feature Extraction (PCA)
        st.subheader("7a. Feature Extraction (PCA)")
        pca_n_components = st.number_input(
            "Number of PCA Components (0 for automatic, 1 for min):",
            min_value=0, value=2, step=1,
            key="pca_n_components"
        )
        apply_pca = st.button("🚀 Apply PCA", key="apply_pca")

        # 7b. Feature Creation
        st.subheader("7b. Feature Creation")
        st.write("Generate new features based on existing ones (e.g., sum, interactions).")
        apply_feature_creation = st.button("🚀 Create Custom Features", key="apply_feature_creation")

        # 7c. Feature Selection
        st.subheader("7c. Feature Selection")
        st.write("Select relevant features, potentially based on correlation with the target variable.")
        
        selection_strategy = st.selectbox(
            "Selection Strategy",
            options=["univariate", "variance", "correlation"],
            index=0,
            key="feature_selection_strategy"
        )
        selection_threshold = st.number_input(
            "Selection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            key="feature_selection_threshold"
        )
        apply_feature_selection = st.button("🚀 Apply Feature Selection", key="apply_feature_selection")

        # ➕ Apply PCA
        if apply_pca:
            st.session_state.data_snapshots['before_feature_engineering_extraction'] = st.session_state.preprocessed_df.copy()
            with st.spinner("Applying PCA for feature extraction..."):
                numeric_df = st.session_state.preprocessed_df.select_dtypes(include=np.number)

                if numeric_df.empty:
                    st.error("No numeric columns found for PCA.")
                else:
                    non_numeric_df = st.session_state.preprocessed_df.select_dtypes(exclude=np.number)
                    pca_extractor = FeatureExtractor(n_components=pca_n_components if pca_n_components > 0 else None)
                    pca_features_df = pca_extractor.fit_transform(numeric_df)

                    st.session_state.preprocessed_df = pd.concat([pca_features_df, non_numeric_df], axis=1)
                    st.session_state.fitted_preprocessing_components['pca_extractor'] = pca_extractor
                    st.session_state.preprocessing_steps_completed['feature_engineering'] = False
                    st.success(f"PCA applied! New features: {', '.join(pca_extractor.extracted_feature_names)}")
                    logger.info("✅ PCA applied.")

        # ➕ Apply Feature Creation
        if apply_feature_creation:
            st.session_state.data_snapshots['before_feature_engineering_creation'] = st.session_state.preprocessed_df.copy()
            with st.spinner("Creating custom features..."):
                feature_creator = FeatureCreator()
                st.session_state.preprocessed_df, new_features = feature_creator.fit_transform(st.session_state.preprocessed_df)

                st.session_state.fitted_preprocessing_components['feature_creator'] = feature_creator
                if new_features:
                    st.session_state.preprocessing_steps_completed['feature_engineering'] = False
                    st.success(f"✅ Custom features created: {', '.join(new_features)}")
                    logger.info(f"New features: {new_features}")
                else:
                    st.info("No new custom features were created.")

        # ➕ Apply Feature Selection
        if apply_feature_selection:
            st.session_state.data_snapshots['before_feature_engineering_selection'] = st.session_state.preprocessed_df.copy()
            with st.spinner("Applying feature selection..."):
                try:
                    feature_selector = FeatureSelector(
                        strategy=selection_strategy,
                        threshold=selection_threshold
                    )

                    df = st.session_state.preprocessed_df.copy()

                    if st.session_state.selected_target and st.session_state.selected_target in df.columns:
                        X = df.drop(columns=[st.session_state.selected_target])
                        y = df[st.session_state.selected_target]
                    else:
                        X = df.copy()
                        y = None

                    X_numeric = X.select_dtypes(include=np.number)

                    # 🔍 Handle missing values
                    if X_numeric.isnull().values.any():
                        imputer = SimpleImputer(strategy='mean')
                        X_numeric = pd.DataFrame(imputer.fit_transform(X_numeric), columns=X_numeric.columns, index=X_numeric.index)
                        if y is not None:
                            y = y.loc[X_numeric.index]

                    if X_numeric.empty:
                        st.error("❌ No numeric columns available for feature selection.")
                    else:
                        selected_df = feature_selector.fit_transform(X_numeric, y)

                        if st.session_state.selected_target and st.session_state.selected_target in df.columns:
                            non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                            combined = pd.concat(
                                [selected_df, df[non_numeric_cols], df[[st.session_state.selected_target]]],
                                axis=1
                            )
                            st.session_state.preprocessed_df = combined
                        else:
                            st.session_state.preprocessed_df = selected_df

                        st.session_state.fitted_preprocessing_components['feature_selector'] = feature_selector
                        st.session_state.preprocessing_steps_completed['feature_engineering'] = True
                        st.success(f"✅ Feature selection complete. {selected_df.shape[1]} features selected.")
                        logger.info("✅ Feature selection completed.")

                except Exception as e:
                    st.error(f"❌ Feature selection failed: {str(e)}")
                    logger.error(f"Feature selection failed: {str(e)}\n{traceback.format_exc()}")

        # 📊 Final Preview
        st.write("**Current Data Preview after Feature Engineering:**")
        st.dataframe(st.session_state.preprocessed_df.head(), use_container_width=True)

    # --- Download and Save Section ---
    st.markdown("---")
    st.subheader("💾 Save & Export")

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create expandable section for export options
    with st.expander("📤 Export Options", expanded=True):
        export_format = st.radio(
            "Select export format:",
            options=["CSV", "Excel", "JSON", "Parquet"],
            horizontal=True
        )

        if export_format == "CSV":
            csv_options = st.columns(2)
            with csv_options[0]:
                csv_index = st.checkbox("Include index", value=False, key="csv_index")
            with csv_options[1]:
                csv_encoding = st.selectbox(
                    "Encoding",
                    options=["utf-8", "latin1", "utf-16"],
                    index=0,
                    key="csv_encoding"
                )

        elif export_format == "Excel":
            excel_options = st.columns(2)
            with excel_options[0]:
                excel_index = st.checkbox("Include index", value=False, key="excel_index")
            with excel_options[1]:
                excel_sheet = st.text_input("Sheet name", value="Sheet1", key="excel_sheet")

        download_filename = st.text_input(
            "Filename",
            value=f"preprocessed_data_{timestamp}.{export_format.lower()}",
            key="download_filename"
        )

        if st.button(f"⬇️ Download as {export_format}", key=f"download_{export_format}"):
            try:
                if export_format == "CSV":
                    csv = st.session_state.preprocessed_df.to_csv(index=csv_index).encode(csv_encoding)
                    st.download_button(
                        label="Confirm Download CSV",
                        data=csv,
                        file_name=download_filename,
                        mime="text/csv",
                        key="confirm_download_csv"
                    )

                elif export_format == "Excel":
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        st.session_state.preprocessed_df.to_excel(
                            writer,
                            sheet_name=excel_sheet,
                            index=excel_index
                        )
                    st.download_button(
                        label="Confirm Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=download_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="confirm_download_excel"
                    )

                elif export_format == "JSON":
                    json_data = st.session_state.preprocessed_df.to_json(orient='records')
                    st.download_button(
                        label="Confirm Download JSON",
                        data=json_data,
                        file_name=download_filename,
                        mime="application/json",
                        key="confirm_download_json"
                    )

                elif export_format == "Parquet":
                    parquet_buffer = BytesIO()
                    st.session_state.preprocessed_df.to_parquet(parquet_buffer)
                    st.download_button(
                        label="Confirm Download Parquet",
                        data=parquet_buffer.getvalue(),
                        file_name=download_filename,
                        mime="application/octet-stream",
                        key="confirm_download_parquet"
                    )

            except Exception as e:
                st.error(f"Export failed: {str(e)}")
                logger.error(f"Export error: {str(e)}\n{traceback.format_exc()}")

        if st.button("📊 Visualize Processed Data (EDA2)"):
            if 'preprocessed_df' in st.session_state and not st.session_state.preprocessed_df.empty:
                st.session_state['raw2_df'] = st.session_state.preprocessed_df.copy()
                st.session_state['current_page'] = 'Visualize'
                st.switch_page("4_Visualize.py")
            else:
                st.warning("No processed data available for visualization. Please complete preprocessing first.")

    # # --- Save Pipeline ---
    # if st.button("💾 Save Preprocessing Pipeline"):
    #     try:
    #         pipeline = {
    #             'missing_value_handler': st.session_state.fitted_preprocessing_components.get('missing_value_handler'),
    #             'outlier_detector': st.session_state.fitted_preprocessing_components.get('outlier_detector'),
    #             'log_transformer': st.session_state.fitted_preprocessing_components.get('log_transformer'),
    #             'feature_scaler': st.session_state.fitted_preprocessing_components.get('feature_scaler'),
    #             'categorical_encoder': st.session_state.fitted_preprocessing_components.get('categorical_encoder'),
    #             'multicollinearity_reducer': st.session_state.fitted_preprocessing_components.get('multicollinearity_reducer'),
    #             'feature_creator': st.session_state.fitted_preprocessing_components.get('feature_creator'),
    #             'feature_selector': st.session_state.fitted_preprocessing_components.get('feature_selector'),
    #             'pca_extractor': st.session_state.fitted_preprocessing_components.get('pca_extractor'),
    #             'settings': {
    #                 'target_column': st.session_state.selected_target,
    #                 'timestamp': timestamp
    #             }
    #         }

    #         pipeline_filename = f"preprocessing_pipeline_{timestamp}.pkl"
    #         with open(pipeline_filename, 'wb') as f:
    #             pickle.dump(pipeline, f)

    #         st.success(f"Pipeline saved successfully as {pipeline_filename}")
    #         logger.info(f"Preprocessing pipeline saved to {pipeline_filename}")

    #         pipeline_buffer = BytesIO()
    #         pickle.dump(pipeline, pipeline_buffer)
    #         pipeline_buffer.seek(0)
    #         st.download_button(
    #             label="⬇️ Download Pipeline (.pkl)",
    #             data=pipeline_buffer,
    #             file_name=pipeline_filename,
    #             mime="application/octet-stream"
    #         )

    #     except Exception as e:
    #         st.error(f"Failed to save pipeline: {str(e)}")
    #         logger.error(f"Error saving pipeline: {str(e)}")

    # --- Advanced Save Format Options ---
    st.markdown("---")
    st.subheader("🔧 Save Preprocessing Pipeline")

    with st.expander("Pipeline Save Options", expanded=True):
        pipeline_format = st.radio(
            "Select pipeline format:",
            options=["PKL (Pickle)", "JSON"],
            horizontal=True,
            key="pipeline_format"
        )

        if pipeline_format == "PKL (Pickle)":
            pkl_options = st.columns(2)
            with pkl_options[0]:
                pkl_protocol = st.selectbox(
                    "Pickle Protocol",
                    options=["Highest (Recommended)", "4", "3"],
                    index=0,
                    key="pkl_protocol"
                )
        elif pipeline_format == "JSON":
            json_options = st.columns(2)
            with json_options[0]:
                json_orient = st.selectbox(
                    "JSON Orientation",
                    options=["records", "split", "index", "columns", "values"],
                    index=1,
                    help="How to arrange the JSON structure"
                )
            with json_options[1]:
                json_indent = st.number_input(
                    "Indentation",
                    min_value=0,
                    max_value=8,
                    value=2,
                    help="Number of spaces for pretty-printing"
                )

        compression = st.selectbox(
            "Compression",
            options=["None", "gzip", "zip", "bz2"],
            index=0,
            help="Compression to apply to saved file"
        )

        pipeline_filename = st.text_input(
            "Pipeline filename",
            value=f"preprocessing_pipeline_{timestamp}",
            key="pipeline_filename"
        )

    if st.button("💾 Save Preprocessing Pipeline", key="save_pipeline"):
        try:
            pipeline = {
                'metadata': {
                    'creation_time': timestamp,
                    'pipeline_format': pipeline_format,
                    'compression': compression,
                    'target_column': st.session_state.selected_target,
                    'transformed_features': st.session_state.get('transformed_features', []),
                    'preprocessing_steps': st.session_state.get('preprocessing_steps_completed', {})
                },
                'components': {
                    'missing_value_handler': st.session_state.fitted_preprocessing_components.get('missing_value_handler'),
                    'outlier_detector': st.session_state.fitted_preprocessing_components.get('outlier_detector'),
                    'log_transformer': st.session_state.fitted_preprocessing_components.get('log_transformer'),
                    'feature_scaler': st.session_state.fitted_preprocessing_components.get('feature_scaler'),
                    'categorical_encoder': st.session_state.fitted_preprocessing_components.get('categorical_encoder'),
                    'multicollinearity_reducer': st.session_state.fitted_preprocessing_components.get('multicollinearity_reducer'),
                    'feature_creator': st.session_state.fitted_preprocessing_components.get('feature_creator'),
                    'feature_selector': st.session_state.fitted_preprocessing_components.get('feature_selector'),
                    'pca_extractor': st.session_state.fitted_preprocessing_components.get('pca_extractor')
                }
            }

            if pipeline_format == "PKL (Pickle)":
                file_ext = ".pkl"
                mime_type = "application/octet-stream"
                protocol = {
                    "Highest (Recommended)": pickle.HIGHEST_PROTOCOL,
                    "4": 4,
                    "3": 3
                }[pkl_protocol]
                full_filename = f"{pipeline_filename}{file_ext}"
                with open(full_filename, 'wb') as f:
                    pickle.dump(pipeline, f, protocol=protocol)

                buffer = BytesIO()
                pickle.dump(pipeline, buffer, protocol=protocol)
                buffer.seek(0)

            else:
                file_ext = ".json"
                mime_type = "application/json"
                json_pipeline = {
                    'metadata': pipeline['metadata'],
                    'components': {}
                }

                for name, component in pipeline['components'].items():
                    if component is not None and hasattr(component, '__sklearn_is_fitted__'):
                        json_pipeline['components'][name] = {
                            'type': str(type(component)),
                            'params': component.get_params(),
                            'fitted': True
                        }
                    else:
                        json_pipeline['components'][name] = component

                full_filename = f"{pipeline_filename}{file_ext}"
                with open(full_filename, 'w') as f:
                    json.dump(json_pipeline, f, indent=json_indent, default=str)

                buffer = BytesIO()
                buffer.write(json.dumps(json_pipeline, indent=json_indent, default=str).encode('utf-8'))
                buffer.seek(0)

            if compression != "None":
                import gzip, zipfile, bz2
                compressed_filename = f"{full_filename}.{compression}"

                if compression == "gzip":
                    with gzip.open(compressed_filename, 'wb') as f:
                        f.write(buffer.getvalue())
                elif compression == "zip":
                    with zipfile.ZipFile(compressed_filename, 'w') as zf:
                        zf.writestr(full_filename, buffer.getvalue())
                elif compression == "bz2":
                    with bz2.open(compressed_filename, 'wb') as f:
                        f.write(buffer.getvalue())

                with open(compressed_filename, 'rb') as f:
                    buffer = BytesIO(f.read())

                full_filename = compressed_filename

            st.success(f"Pipeline saved successfully as {full_filename}")
            logger.info(f"Preprocessing pipeline saved to {full_filename}")
            st.download_button(
                label=f"⬇️ Download Pipeline ({file_ext})",
                data=buffer,
                file_name=full_filename,
                mime=mime_type,
                key=f"download_pipeline_{timestamp}"
            )

        except Exception as e:
            st.error(f"Failed to save pipeline: {str(e)}")
            logger.error(f"Error saving pipeline: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()