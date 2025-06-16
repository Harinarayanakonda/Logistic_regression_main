import streamlit as st
import pandas as pd
import os

from src.core.preprocessing.missing_values import MissingValueHandler
from src.core.preprocessing.outlier_detection import OutlierDetector
from src.core.preprocessing.log_transform import LogTransformer
from src.core.preprocessing.scaling import FeatureScaler
from src.core.preprocessing.categorical_encoding import CategoricalEncoder
from src.core.preprocessing.multicollinearity import MulticollinearityReducer
from src.core.feature_engineering.feature_engineering import FeatureExtractor, FeatureCreator, FeatureSelector
from src.utils.logging import logger
from src.utils.helpers import generate_timestamped_filename
from src.config.settings import AppSettings
from src.config.paths import PROCESSED_DATA_DIR
from src.interfaces.streamlit_app.utils import download_dataframe, intelligent_impute

logger.info("Loading Preprocessing page.")

st.markdown("# 🛠️ Data Preprocessing Pipeline")
st.write("Follow the steps below to preprocess your data. Each step must be run in order, matching the professional pipeline diagram.")

if 'raw1_df' not in st.session_state or st.session_state['raw1_df'] is None:
    st.warning("Please upload a dataset on the 'Upload Dataset' page first.")
    st.stop()

if 'preprocessed_df' not in st.session_state:
    st.session_state['preprocessed_df'] = st.session_state['raw1_df'].copy()

df = st.session_state['preprocessed_df']

# ------------------------------
# Target Variable Selection
# ------------------------------
st.markdown("---")
st.subheader("🎯 Select Target Variable")

columns = df.columns.tolist()
default_target = (
    AppSettings.TARGET_COLUMN
    if hasattr(AppSettings, "TARGET_COLUMN") and AppSettings.TARGET_COLUMN in columns
    else columns[0]
)
if (
    'selected_target' not in st.session_state
    or st.session_state['selected_target'] not in columns
):
    st.session_state['selected_target'] = default_target

target_col = st.selectbox(
    "Select the target variable for your ML task:",
    options=columns,
    index=columns.index(st.session_state['selected_target'])
)
st.session_state['selected_target'] = target_col

# ------------------------------
# Missing Value Handling
# ------------------------------
st.markdown("---")
with st.expander("1️⃣ Handle Missing Values", expanded=True):
    st.write("Detect and handle missing values in your dataset (deletion/imputation).")

    # Separate features and target
    features_df = df.drop(columns=[target_col], errors='ignore')
    target_df = df[[target_col]].copy()

    # Display missing summary
    missing_pct = features_df.isnull().mean() * 100
    missing_flag = features_df.isnull().any()
    missing_summary = pd.DataFrame({
        "Column": features_df.columns,
        "% Missing": missing_pct.round(2),
        "Has Missing?": missing_flag.map({True: "✅", False: "❌"}).values,
        "Category": [
            "<5%" if pct < 5 else "5–30%" if pct <= 30 else ">30%" for pct in missing_pct.values
        ]
    })

    st.write("**Missing Value Summary:**")
    st.dataframe(missing_summary[missing_summary["Has Missing?"] == "✅"], use_container_width=True)

    st.markdown("""
<span style='color:green'><b><5% missing</b></span>: Simple imputation (mean/median/mode)<br>
<span style='color:orange'><b>5–30% missing</b></span>: Advanced imputation (KNN/model-based)<br>
<span style='color:red'><b>>30% missing</b></span>: Will be dropped unless critical
    """, unsafe_allow_html=True)

    # Flag rows with any missing value
    if st.button("Flag Missing Values", key="btn_flag_missing"):
        flagged_features_df = features_df.copy()
        flagged_features_df['Missing_Flag'] = features_df.isnull().any(axis=1).astype(int)
        df_flagged = pd.concat([flagged_features_df, target_df], axis=1)
        st.session_state['preprocessed_df'] = df_flagged
        st.success(f"✅ Flagged {flagged_features_df['Missing_Flag'].sum()} rows with missing values.")
        st.dataframe(df_flagged.head(), use_container_width=True)

    # Run intelligent imputation
    if st.button("Run Intelligent Missing Value Handler", key="btn_missing"):
        try:
            features_for_impute = features_df.drop(columns=['Missing_Flag'], errors='ignore')
            cleaned_features_df, summary_table, imputation_log = intelligent_impute(features_for_impute)

            if 'Missing_Flag' in features_df.columns:
                cleaned_features_df['Missing_Flag'] = features_df['Missing_Flag']

            flagged_rows = features_df['Missing_Flag'].sum() if 'Missing_Flag' in features_df.columns else features_df.isnull().any(axis=1).sum()
            imputed_cols = summary_table[summary_table["Imputation Method"].str.contains("imputation", na=False)]["Column"].tolist()
            dropped_cols = summary_table[summary_table["Imputation Method"].str.contains("dropped", na=False)]["Column"].tolist()

            if cleaned_features_df.empty:
                st.error("All features were dropped during missing value handling.")
                st.stop()

            df_cleaned = pd.concat([cleaned_features_df, target_df], axis=1)
            st.session_state['preprocessed_df'] = df_cleaned

            st.success("✅ Missing values handled successfully!")
            st.markdown(f"- **Rows flagged as missing:** {flagged_rows}")
            st.markdown(f"- **Columns imputed:** {', '.join(imputed_cols) if imputed_cols else 'None'}")
            st.markdown(f"- **Columns dropped:** {', '.join(dropped_cols) if dropped_cols else 'None'}")

            st.write("**Imputation Summary Table:**")
            st.dataframe(summary_table, use_container_width=True)

            st.write("**Imputation Log:**")
            st.code('\n'.join(imputation_log))
            st.info(f"Columns after imputation: {df_cleaned.columns.tolist()}")

        except Exception as e:
            st.error(f"🚫 Missing value handling failed: {e}")

    # Preview cleaned data
    if not st.session_state['preprocessed_df'].empty:
        st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)

# 2️⃣ Outlier Detection & Removal
with st.expander("2️⃣ Outlier Detection & Removal", expanded=False):
    st.write("Detect and handle outliers using IQR or Z-score methods. Preview, flag, or remove outliers from your dataset.")

    # 1️⃣ Detect Outliers
    if st.button("🔍 Detect Outliers", key="btn_outlier_detect"):
        try:
            df = st.session_state['preprocessed_df']
            target_col = st.session_state['selected_target']
            features_df = df.drop(columns=[target_col], errors='ignore')

            detector = OutlierDetector()
            detector.fit(features_df)
            outlier_table = detector.detect_outliers_table(features_df)

            st.session_state['outlier_detector'] = detector
            st.session_state['outlier_table'] = outlier_table

            st.write("📊 **Outlier Table** (True = Outlier):")
            st.dataframe(outlier_table, use_container_width=True)
            st.success("✅ Outlier detection completed!")
            st.info(f"Columns checked: {features_df.columns.tolist()}")
        except Exception as e:
            st.error(f"🚫 Outlier detection failed: {e}")
            st.dataframe(df.head(), use_container_width=True)

    # 2️⃣ Flag Outliers
    if st.button("🚩 Flag Outliers", key="btn_outlier_flag"):
        try:
            df = st.session_state['preprocessed_df']
            detector = st.session_state.get('outlier_detector', None)

            if detector is None:
                st.warning("⚠️ Please run 'Detect Outliers' first.")
            else:
                df_flagged = detector.transform(df)
                st.session_state['preprocessed_df'] = df_flagged
                st.success("🚩 Outlier flag column added successfully!")
                st.dataframe(df_flagged.head(), use_container_width=True)
        except Exception as e:
            st.error(f"🚫 Flagging outliers failed: {e}")
            st.dataframe(df.head(), use_container_width=True)

    # 3️⃣ Remove Outliers
    if st.button("❌ Remove Outliers", key="btn_outlier_remove"):
        try:
            df = st.session_state['preprocessed_df']
            detector = st.session_state.get('outlier_detector', None)

            if detector is None:
                st.warning("⚠️ Please run 'Detect Outliers' first.")
            else:
                df_cleaned = detector.remove_outliers(df)
                st.session_state['preprocessed_df'] = df_cleaned
                st.success("✅ Outliers removed from the dataset.")
                st.dataframe(df_cleaned.head(), use_container_width=True)
        except Exception as e:
            st.error(f"🚫 Outlier removal failed: {e}")
            st.dataframe(df.head(), use_container_width=True)
            



# 3️⃣ Log Transformation
with st.expander("3️⃣ Log Transform Numeric Features", expanded=False):
    st.write("""
    Apply **log transformation** to reduce skewness in numeric features.
    This is especially useful when the data is right-skewed (long tail on the right).
    """)

    # Prepare features (exclude target column)
    feature_cols = [col for col in df.columns if col != target_col]
    features_df = df[feature_cols]
    target_df = df[[target_col]]

    if st.button("🧮 Run Log Transformer", key="btn_log_transform"):
        try:
            transformer = LogTransformer()
            features_df = transformer.fit_transform(features_df)

            # Combine transformed features with target
            df = pd.concat([features_df, target_df], axis=1)
            st.session_state['preprocessed_df'] = df

            st.success("✅ Log transformation applied successfully!")
            st.info(f"Transformed Columns: {features_df.columns.tolist()}")
        except Exception as e:
            st.error(f"🚫 Log transformation failed: {e}")

    st.write("🧾 Updated Data Preview:")
    st.dataframe(df.head(), use_container_width=True)

# 4️⃣ Normalize / Standardize Features
with st.expander("4️⃣ Normalize / Standardize Features", expanded=False):
    st.write("""
    Scale numeric features using **Standard Scaling** or **Min-Max Normalization**  
    to improve model convergence and performance.
    Typically applied after log transformation to normalized features.
    """)

    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    features_df = df[feature_cols]
    target_df = df[[target_col]]

    if st.button("⚖️ Run Feature Scaler", key="btn_scaler"):
        try:
            # Automatically identify log-transformed columns for scaling
            columns_to_scale = [col for col in features_df.columns if col.endswith("_log")]

            if not columns_to_scale:
                st.warning("⚠️ No log-transformed columns found for scaling. Please run the log transformation step first.")
            else:
                scaler = FeatureScaler()
                features_df = scaler.fit_transform(features_df, columns_to_scale)

                # Reattach the target column
                df = pd.concat([features_df, target_df], axis=1)
                st.session_state['preprocessed_df'] = df

                st.success("✅ Feature scaling applied successfully!")
                st.info(f"Scaled columns: {columns_to_scale}")
                st.info(f"Columns after scaling: {df.columns.tolist()}")
        except Exception as e:
            st.error(f"🚫 Feature scaling failed: {e}")

    st.write("📊 Scaled Data Preview:")
    st.dataframe(df.head(), use_container_width=True)


# 5️⃣ Encode Categorical Variables
with st.expander("5️⃣ Encode Categorical Variables", expanded=False):
    st.write("""
    Convert categorical features into numeric format using techniques like **Label Encoding**, **One-Hot Encoding**, or **Ordinal Encoding**.
    This ensures compatibility with machine learning models.
    """)

    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    features_df = df[feature_cols]
    target_df = df[[target_col]]

    if st.button("🔤 Run Categorical Encoder", key="btn_categorical"):
        try:
            encoder = CategoricalEncoder()
            features_df = encoder.fit_transform(features_df)

            # Reattach target column
            df = pd.concat([features_df, target_df], axis=1)
            st.session_state['preprocessed_df'] = df

            st.success("✅ Categorical variables encoded successfully!")
            st.info(f"Total columns after encoding: {len(df.columns)}")

            # Detect any remaining non-numeric columns
            non_numeric_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
            if non_numeric_cols:
                st.warning(f"⚠️ Warning: These columns are still non-numeric after encoding: {non_numeric_cols}")
            else:
                st.info("🎉 All features are now numeric. Data is model-ready.")

            # Show data types
            st.write("📊 Column data types after encoding:")
            st.dataframe(features_df.dtypes.reset_index().rename(columns={"index": "Feature", 0: "DataType"}))

        except Exception as e:
            st.error(f"🚫 Categorical encoding failed: {e}")

    st.write("🔎 Encoded Data Preview:")
    st.dataframe(df.head(), use_container_width=True)

# 6️⃣ Multicollinearity Reduction
with st.expander("6️⃣ Multicollinearity Reduction", expanded=False):
    st.write("""
    Remove highly correlated features using **VIF (Variance Inflation Factor)** or **correlation matrix** to prevent redundant information and improve model generalization.
    """)

    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    features_df = df[feature_cols]
    target_df = df[[target_col]]

    if st.button("📉 Run Multicollinearity Reducer", key="btn_multicol"):
        try:
            # Track columns before reduction
            cols_before = set(features_df.columns)

            # Apply reducer
            reducer = MulticollinearityReducer()
            features_df = reducer.fit_transform(features_df)

            # Recombine with target
            df = pd.concat([features_df, target_df], axis=1)
            st.session_state['preprocessed_df'] = df

            st.success("✅ Multicollinearity reduced successfully!")
            st.info(f"📊 Total columns after reduction: {len(df.columns)}")

            # Show dropped features
            cols_after = set(features_df.columns)
            dropped_cols = list(cols_before - cols_after)
            if dropped_cols:
                st.write("### ❌ Features Dropped Due to High Correlation")
                st.dataframe(pd.DataFrame({"Dropped Feature": dropped_cols}), use_container_width=True)
            else:
                st.info("👍 No features were dropped. No strong multicollinearity detected.")

        except Exception as e:
            st.error(f"🚫 Multicollinearity reduction failed: {e}")

    st.write("🔍 Data Preview After Multicollinearity Reduction:")
    st.dataframe(df.head(), use_container_width=True)


# 7️⃣ Feature Engineering Section
with st.expander("7️⃣ Feature Engineering", expanded=False):
    st.write("""
    Perform advanced feature engineering:
    - **Feature Extraction** (e.g., PCA)
    - **Feature Creation** (domain-specific, synthetic)
    - **Feature Selection** (filter/wrapper/embedded methods)
    """)

    # Prepare features and target
    feature_cols = [col for col in df.columns if col != target_col]
    features_df = df[feature_cols]
    target_df = df[[target_col]]

    col1, col2, col3 = st.columns(3)
    st.dataframe(df.head(), use_container_width=True)

    # Feature Extraction (e.g., PCA)
    with col1:
        if st.button("Feature Extraction (PCA)", key="btn_feature_extraction"):
            try:
                extractor = FeatureExtractor(n_components=2)
                extracted_df = extractor.fit_transform(features_df)
                extracted_df.columns = [f"PC{i+1}" for i in range(extracted_df.shape[1])]
                df = pd.concat([extracted_df, target_df], axis=1)
                st.session_state['preprocessed_df'] = df
                st.success("✅ PCA components added.")
                st.info(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                st.error(f"🚫 Feature extraction failed: {e}")
                st.dataframe(df.head(), use_container_width=True)

    # Feature Creation (e.g., Synthetic Features)
    with col2:
        if st.button("Feature Creation", key="btn_feature_creation"):
            try:
                creator = FeatureCreator()
                created_df = creator.fit_transform(features_df)
                df = pd.concat([created_df, target_df], axis=1)
                st.session_state['preprocessed_df'] = df
                st.success("✅ Synthetic features created.")
                st.info(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                st.error(f"🚫 Feature creation failed: {e}")
                st.dataframe(df.head(), use_container_width=True)

    # Feature Selection (filter methods like chi2, mutual_info)
    with col3:
        if st.button("Feature Selection", key="btn_feature_selection"):
            try:
                numeric_cols = features_df.select_dtypes(include=[float, int]).columns.tolist()
                if len(numeric_cols) < 2:
                    st.warning("⚠️ Not enough numeric columns for selection. At least 2 required.")
                else:
                    selector = FeatureSelector(k=len(numeric_cols))
                    selected_df = selector.fit_transform(features_df[numeric_cols], target_df[target_col])
                    # Combine selected features with original non-numeric and target
                    non_numeric = features_df.drop(columns=numeric_cols)
                    df = pd.concat([selected_df, non_numeric, target_df], axis=1)
                    st.session_state['preprocessed_df'] = df
                    st.success(f"✅ Selected {selected_df.shape[1]} features.")
                    st.info(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                st.error(f"🚫 Feature selection failed: {e}")
                st.dataframe(df.head(), use_container_width=True)



    
# --- Preview and Save Section ---
st.markdown("---")
st.markdown("### 📋 Preview of Current Preprocessed Data")
st.dataframe(st.session_state['preprocessed_df'].head(), use_container_width=True)

st.markdown("---")
st.subheader("💾 Save Processed Data")
if st.button("Save Processed Data", key="btn_save_data"):
    try:
        processed_df = st.session_state['preprocessed_df']
        filename = generate_timestamped_filename("processed_data", "csv")
        processed_df.to_csv(os.path.join(PROCESSED_DATA_DIR, filename), index=False)
        st.session_state['raw2_df'] = processed_df
        st.session_state['preprocessing_success'] = True
        st.success("✅ Data saved successfully!")
        logger.info(f"Data saved to {os.path.join(PROCESSED_DATA_DIR, filename)}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}", exc_info=True)
        st.error(f"🚫 Saving failed: {e}")
        st.session_state['preprocessing_success'] = False
        st.dataframe(df.head(), use_container_width=True)

# --- Results Section ---
st.markdown("---")
if st.session_state.get('preprocessing_success'):
    st.subheader("Preprocessing Results")
    processed_df = st.session_state.get('raw2_df')

    if processed_df is not None:
        st.dataframe(processed_df.head(), use_container_width=True)

        col_dl, col_nav = st.columns(2)
        with col_dl:
            st.write("#### Download Processed Data")
            fmt = st.radio("Choose format:", ("csv", "pkl"), key="download_format_radio")
            download_dataframe(processed_df, "raw2_processed_data", fmt)
        with col_nav:
            st.write("#### Continue to Visualization")
            if st.button("Go to Visualization (EDA2)", key="go_to_eda2_button"):
                st.session_state['current_page'] = 'Visualization'
                st.switch_page("pages/4_Visualize.py")
    else:
        st.error("Processed DataFrame not found.")
        if st.button("Retry Preprocessing", key="retry_preprocessing_button_success_path"):
            st.session_state['preprocessing_success'] = False
            st.rerun()

elif 'preprocessing_success' in st.session_state and not st.session_state['preprocessing_success']:
    st.error("❌ Preprocessing failed previously.")
    if st.button("Retry Preprocessing", key="retry_preprocessing_button_fail_path"):
        st.session_state['preprocessing_success'] = False
        st.rerun()
