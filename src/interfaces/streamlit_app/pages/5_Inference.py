import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.interfaces.streamlit_app.utils import load_data_from_uploader, download_dataframe
from src.utils.logging import logger
from src.config.settings import AppSettings

logger.info("Loading Inference page.")

st.markdown("# User Inference Module")
st.write("Make predictions using the deployed model. Supports bulk dataset upload or single-instance manual input.")

# --- Model selection ---
model_dir = os.path.join("src", "models")
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
if not model_files:
    st.error("No models found in src/models/. Please ask admin to train and save a model.")
    st.stop()

selected_model_file = st.selectbox("Select a model for prediction", model_files, key="model_select")
selected_model_path = os.path.join(model_dir, selected_model_file)

# Robust loading: handle both old and new model formats
with open(selected_model_path, "rb") as f:
    loaded_obj = pickle.load(f)

if isinstance(loaded_obj, dict) and "model" in loaded_obj and "feature_names" in loaded_obj:
    model = loaded_obj["model"]
    feature_names = loaded_obj["feature_names"]
    st.success(f"Loaded model: {selected_model_file}")
else:
    st.error(
        f"Selected model file `{selected_model_file}` is not in the expected format.\n\n"
        "Please retrain and save the model using the following code in your training page:\n\n"
        "model_artifact = {\n"
        "    'model': model,\n"
        "    'feature_names': list(X.columns)\n"
        "}\n"
        "with open(model_path, 'wb') as f:\n"
        "    pickle.dump(model_artifact, f)\n"
    )
    st.stop()

# --- Choose inference mode ---
mode = st.radio(
    "Choose prediction mode:",
    ["Bulk Dataset Upload", "Single Instance Manual Input"],
    key="inference_mode"
)

if mode == "Bulk Dataset Upload":
    st.subheader("Bulk Prediction")
    st.write("Upload a dataset to get predictions for multiple instances.")

    col_bulk1, col_bulk2 = st.columns([1, 2])
    with col_bulk1:
        bulk_file_format = st.selectbox(
            "Select File Format",
            ("csv", "xlsx", "json"),
            key="bulk_file_format"
        )
    with col_bulk2:
        bulk_uploaded_file = st.file_uploader(
            "Upload your dataset for bulk prediction",
            type=["csv", "xlsx", "json"],
            key="bulk_uploader"
        )

    if bulk_uploaded_file:
        st.subheader("Data Preview (First 5 Rows)")
        try:
            preview_df = load_data_from_uploader(bulk_uploaded_file, bulk_file_format)
            if preview_df is not None:
                st.dataframe(preview_df.head())
                st.session_state['bulk_input_df'] = preview_df
            else:
                st.error("Could not preview the uploaded bulk data.")
                if 'bulk_input_df' in st.session_state: del st.session_state['bulk_input_df']
        except Exception as e:
            st.error(f"Error previewing file: {e}")
            if 'bulk_input_df' in st.session_state: del st.session_state['bulk_input_df']

        if 'bulk_input_df' in st.session_state:
            if st.button("Run Bulk Prediction", key="run_bulk_pred"):
                try:
                    input_df = st.session_state['bulk_input_df']
                    missing_cols = [col for col in feature_names if col not in input_df.columns]
                    if missing_cols:
                        st.error(f"Missing columns in input: {missing_cols}")
                    elif input_df.isnull().any().any():
                        st.error("Bulk input contains missing values. Please fill or impute missing values before prediction.")
                    else:
                        input_df = input_df[feature_names]  # Ensure correct order

                        # Check for non-numeric columns
                        non_numeric_cols = input_df.select_dtypes(include=['object', 'category']).columns.tolist()
                        if non_numeric_cols:
                            st.error(f"Non-numeric columns found: {non_numeric_cols}. Please ensure all features are properly encoded as numbers before prediction.")
                        else:
                            preds = model.predict(input_df)
                            pred_df = input_df.copy()
                            pred_df['Prediction'] = preds
                            st.subheader("Predictions")
                            st.dataframe(pred_df)
                            download_dataframe(pred_df, "bulk_predictions", "csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

else:
    st.subheader("Single Instance Prediction")
    st.write("Enter the values for each feature below to predict the target for a single instance. Please ensure all values are filled in.")

    # Helper to make feature names more readable
    def prettify_feature_name(name):
        name = name.replace("_log_scaled", " (log-scaled)")
        name = name.replace("_log", " (log)")
        name = name.replace("_scaled", " (scaled)")
        name = name.replace("_", " ")
        return name.title()

    # Split features into two columns for better UI
    col1, col2 = st.columns(2)
    input_data = {}
    for idx, col in enumerate(feature_names):
        label = prettify_feature_name(col)
        help_text = f"Enter value for {label}."
        # Choose input type based on feature name (customize as needed)
        if "area" in col.lower() or "sf" in col.lower() or "sqft" in col.lower():
            default = 0.0
            input_func = st.number_input
        else:
            default = ""
            input_func = st.text_input
        # Alternate columns
        with (col1 if idx % 2 == 0 else col2):
            val = input_func(label, value=default, help=help_text, key=f"single_{col}")
        input_data[col] = val

    st.markdown("---")
    st.info(
        "ℹ️ **Tip:** Refer to your dataset's summary statistics for typical values. All fields are required."
    )

    if st.button("Predict", key="single_predict"):
        try:
            input_df = pd.DataFrame([input_data])
            # Try to convert all columns to numeric, but keep track of failures
            converted_df = input_df.apply(pd.to_numeric, errors='coerce')
            non_numeric_cols = [col for col in input_df.columns if input_df[col].dtype == object and converted_df[col].isnull().any()]
            if converted_df.isnull().any().any() or non_numeric_cols:
                st.error(f"Please fill in all fields with valid numeric values. Non-numeric or missing values found in: {non_numeric_cols}")
            else:
                pred = model.predict(converted_df)[0]
                st.success(f"Prediction: {pred}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")