import streamlit as st
import pandas as pd
from joblib import load
import traceback
import os

# App Title
st.set_page_config(page_title="Logistic Regression Inference", layout="centered")
st.title("Predict Using Logistic Regression Model")

# Load model pipeline
MODEL_DIR = "src/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]

if not model_files:
    st.error("No pipeline models found in 'src/models/'. Please train and save a pipeline model first.")
    st.stop()

selected_model_file = st.selectbox("Select a trained pipeline model", model_files)
MODEL_PATH = os.path.join(MODEL_DIR, selected_model_file)

@st.cache_resource(show_spinner="Loading model...")
def load_pipeline(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return load(path)

try:
    pipeline = load_pipeline(MODEL_PATH)
    st.success(f"Loaded pipeline model: **{selected_model_file}**")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Prediction mode
mode = st.radio("Choose prediction mode:", ["Single Instance Input", "Bulk Dataset Upload"])

# --- Single Instance Prediction ---
if mode == "Single Instance Input":
    with st.form("user_input_form"):
        st.subheader("Enter Feature Values")

        # Example form fields - replace with actual features used in training
        age = st.number_input("Age", min_value=0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        income = st.number_input("Monthly Income", min_value=0.0)

        submit = st.form_submit_button("Predict")

    if submit:
        try:
            input_df = pd.DataFrame([{
                "age": age,
                "gender": gender,
                "income": income
            }])

            st.write("Input Data:")
            st.dataframe(input_df)

            prediction = pipeline.predict(input_df)[0]
            probas = pipeline.predict_proba(input_df)[0] if hasattr(pipeline, "predict_proba") else None

            st.success(f"Predicted Class: `{prediction}`")
            if probas is not None:
                st.write("Prediction Probabilities:")
                st.dataframe(pd.DataFrame([probas], columns=pipeline.classes_))

        except Exception as e:
            st.error("Prediction failed.")
            st.code(traceback.format_exc())

# --- Bulk Dataset Prediction ---
elif mode == "Bulk Dataset Upload":
    st.subheader("Bulk Prediction")
    bulk_file = st.file_uploader("Upload CSV file with input data", type=["csv"])

    if bulk_file:
        try:
            input_df = pd.read_csv(bulk_file)
            st.write("Uploaded Data:")
            st.dataframe(input_df.head())

            predictions = pipeline.predict(input_df)
            output_df = input_df.copy()
            output_df["Prediction"] = predictions

            st.subheader("Predictions")
            st.dataframe(output_df)

            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Failed to make predictions: {e}")
            st.code(traceback.format_exc())