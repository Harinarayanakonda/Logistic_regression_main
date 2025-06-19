import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# Assuming these imports are correct based on your project structure
from src.core.preprocessing.pipeline_orchestrator import PreprocessingPipelineOrchestrator
from src.utils.logging import logger
from src.config.settings import AppSettings  # For TARGET_COLUMN
from src.interfaces.streamlit_app.utils import load_data_from_uploader, download_dataframe  # Assuming these utilities exist

logger.info("Loading Prediction page.")

st.markdown("# 🚀 Make Predictions")
st.write("Use the loaded model and preprocessing pipeline to make predictions. Supports bulk dataset upload or single-instance manual input.")

# --- Load the latest preprocessed or raw data ---
df = st.session_state.get('preprocessed_df', st.session_state.get('raw_df'))
if df is None:
    st.error("❌ No data found. Please upload and preprocess your dataset first.")
    st.stop()

# --- Load Model and Preprocessing Pipeline ---
model_dir = os.path.join("src", "models")
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
if not model_files:
    st.error("No models found in src/models/. Please ask admin to train and save a model.")
    st.stop()

selected_model_file = st.selectbox("Select a model for prediction", model_files, key="model_select")
selected_model_path = os.path.join(model_dir, selected_model_file)

model = None
pipeline = None # This will hold the loaded PreprocessingPipelineOrchestrator instance
# target_column = AppSettings.TARGET_COLUMN # This will now be user-selected

try:
    # --- CRITICAL FIX/CHECK: Ensure the pipeline loads correctly ---
    # The error 'NoneType' object has no attribute 'transform' often means 'pipeline' is None here.
    pipeline = PreprocessingPipelineOrchestrator.load("artifacts/preprocessing_pipeline.pkl")
    if pipeline is None:
        raise ValueError("Preprocessing pipeline failed to load and returned None.")

    with open(selected_model_path, "rb") as f:
        loaded_obj = pickle.load(f)

    if isinstance(loaded_obj, dict) and "model" in loaded_obj and "feature_names" in loaded_obj:
        model = loaded_obj["model"]
        model_expected_features = loaded_obj["feature_names"]
        st.success(f"Loaded model and preprocessing pipeline successfully: **{selected_model_file}**")
    else:
        st.error(
            f"Selected model file `{selected_model_file}` is not in the expected format.\n\n"
            "Please retrain and save the model using the following code in your training page:\n\n"
            "```python\n"
            "model_artifact = {\n"
            "  'model': model,\n"
            "  'feature_names': list(X.columns) # X should be the preprocessed features used for training\n"
            "}\n"
            "with open(model_path, 'wb') as f:\n"
            "    pickle.dump(model_artifact, f)\n"
            "```\n"
        )
        st.stop()

    # The predict_with_preprocessing function will now need the list of independent features
    def predict_with_preprocessing(user_raw_df_instance):
        # The pipeline expects all raw features that were present during its fit,
        # so ensure user_raw_df_instance has all those, even if some are NaN.
        # This is crucial for consistent preprocessing.
        processed = pipeline.transform(user_raw_df_instance.copy())

        # Now, create the final input for the model using 'model_expected_features'
        # which are the features the *model* itself was trained on (post-preprocessing).
        final_input_for_model = pd.DataFrame(0.0, index=processed.index, columns=model_expected_features)
        for col in model_expected_features:
            if col in processed.columns:
                final_input_for_model[col] = processed[col]
            # Handle cases where a model_expected_feature might not be in processed
            # (e.g., if it was a target in training, or dropped by pipeline, but still expected by the model)
            # This logic needs to be very robust and match your exact preprocessing
            # and model training feature set.
        return model.predict(final_input_for_model)

except Exception as e:
    st.error(f"❌ Failed to load model or preprocessing pipeline: {e}")
    logger.error(f"Failed to load model or preprocessing pipeline: {e}", exc_info=True)
    st.stop() # Stop execution if loading fails

# --- Additional check to ensure pipeline is not None after the try-except block ---
if pipeline is None:
    st.error("❌ Preprocessing pipeline is not loaded. Cannot proceed with predictions.")
    st.stop()
if model is None:
    st.error("❌ Model is not loaded. Cannot proceed with predictions.")
    st.stop()


try:
    # Get ALL raw feature names from the pipeline that it expects as input
    # This list should include the *original* target column if it was part of the raw data.
    all_raw_feature_names = pipeline.get_raw_feature_names()
    if not all_raw_feature_names:
        st.error("❌ Could not retrieve raw feature names from the preprocessing pipeline. Ensure the pipeline is correctly fitted and saved.")
        st.stop()

    st.session_state['all_raw_feature_names'] = all_raw_feature_names
    st.session_state['final_model_feature_names'] = model_expected_features

except Exception as e:
    st.error(f"❌ Error determining original input feature names from the pipeline: {e}")
    st.stop()

mode = st.radio(
    "Choose prediction mode:",
    ("Single Instance Manual Input", "Bulk Dataset Upload"),
    key="inference_mode"
)

# --- Single Instance Prediction ---
if mode == "Single Instance Manual Input":
    st.subheader("Single Instance Prediction")
    st.write("Enter the values for each feature below to predict the target for a single instance. Please ensure all values are filled in.")

    # --- NEW: Target Variable Selection Dropdown ---
    # Filter out any known internal flag columns or specific columns you never want as a target
    # For robust production apps, you might store "valid_target_columns" in a config
    internal_flag_columns = ['Missing', 'Outlier'] # Define here to avoid repetition
    features_to_exclude_from_target_options = internal_flag_columns + [
        col for col in st.session_state['all_raw_feature_names'] if isinstance(col, str) and ('_Flag' in col or '_flag' in col)
    ]

    potential_target_options = [
        col for col in st.session_state['all_raw_feature_names']
        if col not in features_to_exclude_from_target_options
    ]

    if not potential_target_options:
        st.error("❌ No valid features found to select as a target. Please check your data and pipeline configuration.")
        st.stop()

    # Get initial target from AppSettings if available, else first valid option
    default_target_idx = 0
    if AppSettings.TARGET_COLUMN and AppSettings.TARGET_COLUMN in potential_target_options:
        default_target_idx = potential_target_options.index(AppSettings.TARGET_COLUMN)

    selected_target_column = st.selectbox(
        "Select the **Target Variable** to predict:",
        potential_target_options,
        index=default_target_idx, # Set default if AppSettings.TARGET_COLUMN is found
        key="selected_target_for_manual_input"
    )

    # --- Determine Independent Features based on selected target ---
    features_to_display_for_input = [
        col for col in st.session_state['all_raw_feature_names']
        if col != selected_target_column and
        col not in internal_flag_columns and # Still filter out internal flags
        not (isinstance(col, str) and ('_Flag' in col or '_flag' in col))
    ]

    if not features_to_display_for_input:
        st.warning(f"No independent features to display after excluding '{selected_target_column}'.")
        # Do not stop if there are no independent features; the user might be experimenting.
        # However, a prediction would not make sense here.
        if st.button("Predict Single Instance (No Features)", key="predict_single_instance_btn_no_features", disabled=True):
             pass # This button will be disabled.
        st.stop() # Stop further execution to prevent trying to build an empty DataFrame.


    input_data = {}
    st.markdown("---")
    st.write(f"### Enter Values for Features (excluding '{selected_target_column}')")
    num_cols = 2
    cols = st.columns(num_cols)

    def prettify_feature_name(name):
        name = name.replace("_", " ")
        return name.title()

    for i, feature in enumerate(features_to_display_for_input):
        with cols[i % num_cols]:
            # Basic type inference for input widget
            # You might want to get actual data types from your pipeline or schema here
            if any(key_word in feature.lower() for key_word in ["area", "sf", "sqft", "year", "yrs", "bedroom", "bath", "garage", "num", "count", "value", "age"]):
                default_val = 0.0
                input_widget = st.number_input
            else:
                default_val = ""
                input_widget = st.text_input # Use text_input for general cases, including categorical

            input_data[feature] = input_widget(
                label=prettify_feature_name(feature),
                value=default_val,
                help=f"Enter value for {prettify_feature_name(feature)}.",
                key=f"input_{feature}_{selected_target_column}" # Add target_column to key for uniqueness on target change
            )

    st.markdown("---")
    st.info("ℹ️ **Tip:** Refer to your dataset's summary statistics for typical values. All fields are required.")

    if st.button("Predict Single Instance", key="predict_single_instance_btn"):
        # 1. Validate all required fields are filled
        all_fields_filled = True
        for feature in features_to_display_for_input:
            if input_data.get(feature) == "" or input_data.get(feature) is None: # Check for empty string or None from widgets
                st.error(f"Please fill in all fields. '{prettify_feature_name(feature)}' is empty.")
                all_fields_filled = False
                break
        if not all_fields_filled:
            st.stop() # Stop execution if validation fails

        # 2. Create a DataFrame that has ALL original raw features, filling missing with NaN
        #    (The pipeline expects all features it was trained on)
        full_raw_input_df_for_pipeline = pd.DataFrame(columns=st.session_state['all_raw_feature_names'])

        # Populate with user inputs
        for col in st.session_state['all_raw_feature_names']:
            if col == selected_target_column:
                full_raw_input_df_for_pipeline[col] = np.nan # Target column should be NaN for prediction
            elif col in input_data:
                # Convert string inputs to numeric if applicable, for robust parsing
                val = input_data[col]
                # Attempt to convert to float, if it fails, keep as string (for categorical)
                try:
                    full_raw_input_df_for_pipeline[col] = float(val) if val != "" else np.nan
                except ValueError:
                    full_raw_input_df_for_pipeline[col] = val
            else:
                # Features not displayed (e.g., flags from your internal_flag_columns list) should be NaN
                full_raw_input_df_for_pipeline[col] = np.nan

        # Ensure all columns are converted to appropriate types if necessary by Pandas
        # This can help avoid issues if different input widgets provide different types
        full_raw_input_df_for_pipeline = full_raw_input_df_for_pipeline.apply(pd.to_numeric, errors='ignore')


        try:
            prediction = predict_with_preprocessing(full_raw_input_df_for_pipeline)

            # For predict_proba, use the original full raw input dataframe again
            # to pass through the pipeline to get the exact processed format for model.predict_proba
            processed_data_for_proba = pipeline.transform(full_raw_input_df_for_pipeline.copy())
            final_input_for_model_for_proba = pd.DataFrame(0.0, index=processed_data_for_proba.index, columns=st.session_state['final_model_feature_names'])
            for col in st.session_state['final_model_feature_names']:
                if col in processed_data_for_proba.columns:
                    final_input_for_model_for_proba[col] = processed_data_for_proba[col]
                else:
                    # If any expected model feature is missing after preprocessing, fill with 0 or NaN
                    # depending on what your model expects for missing processed features.
                    # Defaulting to 0.0 as per your final_input_for_model initialization
                    final_input_for_model_for_proba[col] = 0.0


            prediction_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    prediction_proba = model.predict_proba(final_input_for_model_for_proba)
                except Exception as proba_e:
                    st.warning(f"Could not get prediction probabilities: {proba_e}")
                    logger.warning(f"Error getting prediction probabilities: {proba_e}")


            st.success("Prediction Results:")
            st.write(f"Predicted **{selected_target_column}**: **{prediction[0]:.4f}**") # Format prediction
            if prediction_proba is not None and len(model.classes_) > 1:
                st.write("Prediction Probabilities:")
                proba_df = pd.DataFrame(prediction_proba, columns=[f"Class {c}" for c in model.classes_])
                st.dataframe(proba_df, use_container_width=True)
            elif prediction_proba is not None and len(model.classes_) == 1:
                st.info("Model is binary classification or regression. Probabilities for single class or regression models are not typically displayed this way.")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            logger.error(f"Error during single instance prediction: {e}", exc_info=True)

# --- Bulk Dataset Upload ---
elif mode == "Bulk Dataset Upload":
    st.subheader("Bulk Prediction")
    st.write("Upload a dataset to get predictions for multiple instances. Ensure it has the same raw features as your training data.")

    col_bulk1, col_bulk2 = st.columns([1, 2])
    with col_bulk1:
        bulk_file_format = st.selectbox("Select File Format", ("csv", "xlsx", "json"), key="bulk_file_format")
    with col_bulk2:
        bulk_uploaded_file = st.file_uploader("Upload your dataset for bulk prediction", type=["csv", "xlsx", "json"], key="bulk_uploader")

    if bulk_uploaded_file:
        st.subheader("Data Preview (First 5 Rows)")
        try:
            raw_bulk_df = load_data_from_uploader(bulk_uploaded_file, bulk_file_format)
            if raw_bulk_df is not None:
                st.dataframe(raw_bulk_df.head(), use_container_width=True)
                st.session_state['bulk_input_df'] = raw_bulk_df
            else:
                st.error("❌ Could not preview the uploaded bulk data.")
                if 'bulk_input_df' in st.session_state:
                    del st.session_state['bulk_input_df']
        except Exception as e:
            st.error(f"❌ Error previewing file: {e}")
            if 'bulk_input_df' in st.session_state:
                del st.session_state['bulk_input_df']

    if 'bulk_input_df' in st.session_state:
        # User needs to select which column is the target in the uploaded bulk file
        # It's important that this matches one of the `all_raw_feature_names`
        st.markdown("---")
        st.write("### Bulk Prediction Configuration")
        bulk_target_col_options = ['None (All columns are features)'] + st.session_state['all_raw_feature_names']
        selected_bulk_target_column = st.selectbox(
            "Is there a **Target Column** in your uploaded file that should be excluded from prediction inputs?",
            bulk_target_col_options,
            index=0, # Default to None
            key="bulk_target_column_select"
        )
        st.info("If your uploaded file contains the actual target column for validation/analysis, select it here. It will be excluded from the model input.")

        if st.button("Run Bulk Prediction", key="run_bulk_pred"):
            try:
                bulk_df_to_transform = st.session_state['bulk_input_df'].copy()

                # If a target column is selected in bulk, drop it before sending to pipeline
                if selected_bulk_target_column != 'None (All columns are features)':
                    if selected_bulk_target_column in bulk_df_to_transform.columns:
                        st.write(f"Excluding selected target column '**{selected_bulk_target_column}**' from bulk prediction input.")
                        bulk_df_to_transform = bulk_df_to_transform.drop(columns=[selected_bulk_target_column])
                    else:
                         st.warning(f"Selected target column '**{selected_bulk_target_column}**' not found in uploaded bulk data. All columns will be used as features.")


                # Align the bulk DataFrame to the expected raw features for the pipeline
                if st.session_state.get('all_raw_feature_names'):
                    # Create an empty DataFrame with all expected raw features
                    aligned_bulk_df = pd.DataFrame(columns=st.session_state['all_raw_feature_names'])

                    # Populate aligned_bulk_df with data from bulk_df_to_transform
                    # Any feature in `all_raw_feature_names` not in `bulk_df_to_transform` will be NaN
                    for col in st.session_state['all_raw_feature_names']:
                        if col in bulk_df_to_transform.columns:
                            aligned_bulk_df[col] = bulk_df_to_transform[col]
                        else:
                            # For the target column (if not explicitly dropped) or other missing features,
                            # fill with NaN so the pipeline can handle imputation consistently.
                            aligned_bulk_df[col] = np.nan

                    bulk_predictions = predict_with_preprocessing(aligned_bulk_df) # No need for independent_features arg here
                else:
                    st.error("❌ Original raw feature names from pipeline not found. Please ensure pipeline is fitted and saved.")
                    st.stop()

                final_output_df = st.session_state['bulk_input_df'].copy()
                final_output_df['Prediction'] = bulk_predictions

                st.subheader("Predictions")
                st.dataframe(final_output_df)
                download_dataframe(final_output_df, "bulk_predictions", "csv")

            except Exception as e:
                st.error(f"❌ Bulk prediction failed: {e}")
                logger.error(f"Error during bulk prediction: {e}", exc_info=True)