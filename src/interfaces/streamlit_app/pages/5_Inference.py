import streamlit as st
import pandas as pd
from joblib import load
import traceback
import json
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit Page Config
st.set_page_config(page_title="Logistic Regression Inference", layout="centered")
st.title("\U0001F50D Predict Using Logistic Regression Model")

# Define Paths
MODEL_DIR = Path("/home/hari/Logistic_regression_main/src/models/Trained_model")
SCHEMA_DIR = Path("/home/hari/Logistic_regression_main/src/models/infrence_features")
PIPELINE_DIR = Path("/home/hari/Logistic_regression_main/src/models/pipeline")
PROCESSED_SCHEMA_PATH = Path("/home/hari/Logistic_regression_main/src/models/preprocessed_features/schema_processed_features.json")

# Ensure directories exist
for directory in [MODEL_DIR, PIPELINE_DIR, SCHEMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Get latest file by extension
def get_latest_file(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*.{ext}"))
    return max(files, key=lambda x: x.stat().st_mtime) if files else None

# Load artifacts
model_file = get_latest_file(MODEL_DIR, ["joblib", "pkl"])
pipeline_file = get_latest_file(PIPELINE_DIR, ["joblib", "pkl"])
schema_file = get_latest_file(SCHEMA_DIR, ["json"])

st.write("\U0001F9E0 Model file found:", model_file)
st.write("⚖️ Pipeline file found:", pipeline_file)
st.write("\U0001F4C4 Raw Schema file found:", schema_file)

if not model_file or not pipeline_file or not schema_file:
    st.error("❌ Missing model, pipeline, or raw schema.")
    st.stop()

# Cache loader
@st.cache_resource(show_spinner="Loading model and pipeline...")
def load_artifact(path):
    return load(path)

# Load everything
try:
    model = load_artifact(model_file)
    pipeline = load_artifact(pipeline_file)

    with open(schema_file, "r") as f:
        schema = json.load(f)
        if isinstance(schema, dict) and "features" in schema:
            schema = schema["features"]
        elif isinstance(schema, list) and isinstance(schema[0], str):
            schema = [{"name": name, "dtype": "float64"} for name in schema]

    with open(PROCESSED_SCHEMA_PATH, "r") as f:
        processed_schema = json.load(f)

    st.success("✅ All artifacts loaded successfully!")

except Exception as e:
    logger.exception("❌ Failed to load artifacts.")
    st.error("Failed to load model/pipeline/schema.")
    st.code(traceback.format_exc())
    st.stop()

# Extract names and dtypes
FEATURES = [f["name"] for f in schema]
DTYPES = {f["name"]: f["dtype"] for f in schema}

def align_features_with_schema(df: pd.DataFrame, expected_schema: list) -> pd.DataFrame:
    feature_names = [f["name"] for f in expected_schema]
    for feature in feature_names:
        if feature not in df.columns:
            dtype = next((f["dtype"] for f in expected_schema if f["name"] == feature), "float64")
            df[feature] = 0 if dtype in ["int64", "int32"] else 0.0 if dtype == "float64" else False
    df = df[feature_names]
    for feature in feature_names:
        dtype = next((f["dtype"] for f in expected_schema if f["name"] == feature), "float64")
        try:
            if dtype in ["int64", "int32"]:
                df[feature] = df[feature].astype(int)
            elif dtype == "bool":
                df[feature] = df[feature].astype(bool)
            elif dtype == "float64":
                df[feature] = df[feature].astype(float)
        except Exception as e:
            st.error(f"Failed to convert feature {feature} to {dtype}")
            raise
    df = df.fillna(0)
    return df

# UI: Mode selection
mode = st.radio("Choose prediction mode:", ["Single Instance Input", "Bulk Dataset Upload"])

def get_feature_input():
    inputs = {}
    for feature in FEATURES:
        dtype = DTYPES.get(feature, "float64")
        if dtype == "bool":
            inputs[feature] = st.selectbox(feature, [0, 1])
        elif dtype in ["int64", "int32"]:
            inputs[feature] = st.number_input(feature, value=0, step=1)
        else:
            inputs[feature] = st.number_input(feature, value=0.0)
    return inputs

# SINGLE INSTANCE
if mode == "Single Instance Input":
    with st.form("prediction_form"):
        st.subheader("\u270D️ Enter Feature Values")
        user_input = get_feature_input()
        submit = st.form_submit_button("Predict")

    if submit:
        try:
            with st.spinner("\U0001F501 Running Prediction..."):
                input_df = pd.DataFrame([user_input])
                st.write("\U0001F4C4 Raw Input")
                st.dataframe(input_df)

                aligned_input = align_features_with_schema(input_df, schema)
                transformed = pipeline.transform(aligned_input)

                processed_df = pd.DataFrame(transformed)
                processed_df = processed_df.replace([np.inf, -np.inf], np.nan).fillna(0)

                if len(processed_schema) != processed_df.shape[1]:
                    st.error("❌ Transformed feature count doesn't match processed schema.")
                    st.stop()

                processed_df.columns = processed_schema
                st.write("✅ Transformed Features")
                st.dataframe(processed_df.head())
                st.write("\U0001F50E NaN Check")
                st.dataframe(processed_df.isna().sum())

                prediction = model.predict(processed_df)[0]
                proba = model.predict_proba(processed_df)[0] if hasattr(model, "predict_proba") else None

            st.success(f"✨ Prediction: `{prediction}`")
            if proba is not None:
                st.write("\U0001F4CA Probabilities:")
                st.dataframe(pd.DataFrame([proba], columns=model.classes_))

        except Exception as e:
            logger.exception("❌ Single prediction failed")
            st.error("Prediction failed:")
            st.code(traceback.format_exc())

# BULK CSV
elif mode == "Bulk Dataset Upload":
    st.subheader("\U0001F4C2 Upload CSV File")
    file = st.file_uploader("Choose CSV", type=["csv"])

    if file:
        try:
            with st.spinner("\U0001F501 Processing file..."):
                df = pd.read_csv(file)
                st.write("\U0001F4C4 Input Sample")
                st.dataframe(df.head())

                aligned_df = align_features_with_schema(df, schema)
                transformed = pipeline.transform(aligned_df)

                processed_df = pd.DataFrame(transformed)
                processed_df = processed_df.replace([np.inf, -np.inf], np.nan).fillna(0)

                if len(processed_schema) != processed_df.shape[1]:
                    st.error("❌ Transformed feature count doesn't match processed schema.")
                    st.stop()

                processed_df.columns = processed_schema

                predictions = model.predict(processed_df)
                proba = model.predict_proba(processed_df) if hasattr(model, "predict_proba") else None

                df_out = df.copy()
                df_out["Prediction"] = predictions

                if proba is not None:
                    for i, label in enumerate(model.classes_):
                        df_out[f"Prob_{label}"] = proba[:, i]

            st.write("\U0001F4C8 Predictions")
            st.dataframe(df_out.head())

            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions",
                csv,
                f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

        except Exception as e:
            logger.exception("❌ Bulk prediction failed")
            st.error("Bulk prediction failed:")
            st.code(traceback.format_exc())

# Debug Info
with st.expander("\U0001F50D Debug Info"):
    st.write("### Model Type")
    st.write(type(model))

    if hasattr(model, "classes_"):
        st.write("### Classes")
        st.write(model.classes_)

    st.write("### Pipeline Steps")
    for name, step in pipeline.named_steps.items():
        st.write(f"{name}: {type(step).__name__}")

    st.write("### Raw Input Features")
    st.write(FEATURES)

    st.write("### Processed Schema")
    st.write(processed_schema)
