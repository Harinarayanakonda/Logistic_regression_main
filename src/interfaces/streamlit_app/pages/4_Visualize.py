import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from io import BytesIO
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.interfaces.streamlit_app.components.visualization import display_eda
from src.utils.logging import logger
from src.config.settings import AppSettings

import torch
import onnx
import torch.nn as nn
import torch.onnx
import tensorflow as tf
from sklearn.base import BaseEstimator

logger.info("Loading Visualize page.")

st.markdown("# 📊 Data Visualization & Comparison")

# Train model
@st.cache_resource(show_spinner=False)
def train_logistic_regression(X, y):
    model = LogisticRegression(
        random_state=getattr(AppSettings, "RANDOM_STATE", 42),
        solver='liblinear'
    )
    model.fit(X, y)
    return model

# Save model in various formats

def save_model_formats(model: BaseEstimator, X_sample: pd.DataFrame, model_name: str = "logistic_model"):
    model_dir = os.path.join("src", "models")
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_versioned = f"{model_name}_{timestamp}"

    results = {}

    # 1. Pickle
    try:
        pkl_path = os.path.join(model_dir, f"{model_name_versioned}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        results['Pickle'] = pkl_path
    except Exception as e:
        results['Pickle'] = f"❌ Failed: {e}"

    # 2. Joblib
    try:
        joblib_path = os.path.join(model_dir, f"{model_name_versioned}.joblib")
        joblib.dump(model, joblib_path)
        results['Joblib'] = joblib_path
    except Exception as e:
        results['Joblib'] = f"❌ Failed: {e}"

    # 3. ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx_path = os.path.join(model_dir, f"{model_name_versioned}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        results['ONNX'] = onnx_path
    except Exception as e:
        results['ONNX'] = f"❌ Failed: {e}"

    # 4. TensorFlow (as Keras H5)
    try:
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(X_sample.shape[1],), activation='sigmoid')
        ])
        keras_model.compile(optimizer='adam', loss='binary_crossentropy')
        tf_path = os.path.join(model_dir, f"{model_name_versioned}.h5")
        keras_model.save(tf_path)
        results['TensorFlow (H5)'] = tf_path
    except Exception as e:
        results['TensorFlow (H5)'] = f"❌ Failed: {e}"

    # 5. PyTorch (dummy model for format illustration)
    try:
        class DummyTorchModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)
            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        torch_model = DummyTorchModel(X_sample.shape[1])
        dummy_input = torch.randn(1, X_sample.shape[1])
        torch_path = os.path.join(model_dir, f"{model_name_versioned}.pt")
        torch.save(torch_model.state_dict(), torch_path)
        results['PyTorch'] = torch_path
    except Exception as e:
        results['PyTorch'] = f"❌ Failed: {e}"

    return results, model_name_versioned

# UI to train and save
st.subheader("🤖 Train & Save Dummy Model")
if 'raw2_df' in st.session_state and st.session_state['raw2_df'] is not None:
    raw2_df = st.session_state['raw2_df']
    target_col = st.session_state.get('selected_target', raw2_df.columns[0])
    if st.button("Train Model", key="btn_train_model"):
        try:
            X = raw2_df.drop(columns=[target_col])
            y = raw2_df[target_col]

            # Show class distribution
            st.write("Class distribution in target:")
            st.write(y.value_counts())

            vc = y.value_counts()
            valid_classes = vc[vc >= 2].index.tolist()
            if len(valid_classes) < 2:
                st.error("❌ Not enough classes with ≥2 samples for stratified split.")
                st.stop()

            mask = y.isin(valid_classes)
            X = X[mask]
            y = y[mask]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y,
                random_state=getattr(AppSettings, "RANDOM_STATE", 42)
            )

            model = train_logistic_regression(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Training complete. Accuracy: {acc:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            params_df = pd.DataFrame({
                'Parameter': model.get_params().keys(),
                'Value': model.get_params().values()
            })
            st.subheader("🔧 Model Parameters")
            st.dataframe(params_df)

            st.session_state["trained_model"] = model
            st.session_state["model_input_sample"] = X_train.head(1)
            st.session_state["model_accuracy"] = acc

        except Exception as e:
            st.error(f"Model training failed: {e}")

    if "trained_model" in st.session_state and st.button("Save Model in Multiple Formats", key="btn_save_model"):
        try:
            results, model_tag = save_model_formats(
                st.session_state["trained_model"],
                st.session_state["model_input_sample"]
            )

            metadata = {
                "trained_on": datetime.now().isoformat(),
                "accuracy": st.session_state["model_accuracy"],
                "params": st.session_state["trained_model"].get_params(),
                "exports": results
            }

            st.success("✅ Model saved in multiple formats:")
            st.table(pd.DataFrame(list(results.items()), columns=["Format", "Path/Status"]))
            st.json(metadata)

        except Exception as e:
            st.error(f"Saving models failed: {e}")
else:
    st.warning("⚠️ No processed data found. Please run preprocessing first.")