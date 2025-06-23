# src/interfaces/streamlit_app/pages/6_Model_Predictor.py

import streamlit as st
import pandas as pd
from joblib import load
import traceback
import numpy as np
import os

# Path to the trained model
MODEL_PATH = "/home/hari/Logistic_regression_main/src/models/Trained_model/model_20250623_070125.pkl"

st.set_page_config(page_title="Model Predictor", layout="wide")

st.title("📈 Model Inference from Trained Model")

# Load the model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found!")
        return None
    try:
        model = load(MODEL_PATH)
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model:\n{e}")
        return None

model = load_model()

# Input method
input_mode = st.radio("Select input mode:", ["Manual Input", "Upload CSV"])

if model:
    input_df = None

    if input_mode == "Manual Input":
        st.subheader("🔢 Enter feature values manually:")
        try:
            # Example: Replace with your actual feature names and types
            feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else ["feature1", "feature2"]
            manual_input = {}
            for feature in feature_names:
                manual_input[feature] = st.number_input(f"Enter {feature}", step=0.1)
            input_df = pd.DataFrame([manual_input])
        except Exception as e:
            st.error("Error generating manual input fields.")
            st.exception(e)

    elif input_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])
        if uploaded_file:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.dataframe(input_df.head())
            except Exception as e:
                st.error("Error reading the uploaded file.")
                st.exception(e)

    # Prediction
    if input_df is not None and st.button("🔍 Predict"):
        try:
            prediction = model.predict(input_df)
            st.success(f"🎯 Prediction Result: {prediction}")
        except Exception as e:
            st.error("Prediction failed.")
            st.text(traceback.format_exc())
