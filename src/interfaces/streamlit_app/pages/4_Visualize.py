import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from io import BytesIO
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, precision_recall_curve, f1_score, roc_curve, auc
from src.interfaces.streamlit_app.components.visualization import display_eda
from src.utils.logging import logger
from src.config.settings import AppSettings

import torch
import onnx
import torch.nn as nn
import torch.onnx
import tensorflow as tf
from sklearn.base import BaseEstimator
from imblearn.over_sampling import SMOTE

logger.info("Loading Visualize page.")

st.markdown("# 📊 Data Visualization, Comparison & Model Training")

# EDA 1: Raw Data
st.subheader("🔍 EDA1: Raw Data")
if 'raw_df' in st.session_state and not st.session_state['raw_df'].empty:
    display_eda(st.session_state['raw_df'], title_suffix="(Raw)", key_prefix="eda_raw")
else:
    st.warning("Raw dataset not found. Please upload or load raw data first.")

# EDA 2: Preprocessed Data
st.subheader("🧪 EDA2: Preprocessed Data")
if 'raw2_df' in st.session_state and not st.session_state['raw2_df'].empty:
    display_eda(st.session_state['raw2_df'], title_suffix="(Processed)", key_prefix="eda_processed")
else:
    st.warning("Preprocessed dataset not found. Please complete preprocessing first.")

# Compare basic summary stats
if 'raw_df' in st.session_state and 'raw2_df' in st.session_state:
    st.subheader("📈 EDA Comparison Summary")
    with st.expander("Compare Raw vs Preprocessed Summary", expanded=False):
        raw_summary = st.session_state['raw_df'].describe(include='all')
        preprocessed_summary = st.session_state['raw2_df'].describe(include='all')

        st.write("**Raw Data Summary:**")
        st.dataframe(raw_summary)

        st.write("**Preprocessed Data Summary:**")
        st.dataframe(preprocessed_summary)

# Train model
@st.cache_resource(show_spinner=False)
def train_logistic_regression(X, y):
    model = LogisticRegression(
        random_state=getattr(AppSettings, "RANDOM_STATE", 42),
        solver='liblinear',
        class_weight='balanced'
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

    try:
        with open(os.path.join(model_dir, f"{model_name_versioned}.pkl"), "wb") as f:
            pickle.dump(model, f)
        results['Pickle'] = os.path.join(model_dir, f"{model_name_versioned}.pkl")
    except Exception as e:
        results['Pickle'] = f"❌ Failed: {e}"

    try:
        joblib_path = os.path.join(model_dir, f"{model_name_versioned}.joblib")
        joblib.dump(model, joblib_path)
        results['Joblib'] = joblib_path
    except Exception as e:
        results['Joblib'] = f"❌ Failed: {e}"

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

# Train & Save Section
st.subheader("🧐 Train Model on Preprocessed Data")
if 'raw2_df' in st.session_state and st.session_state['raw2_df'] is not None:
    raw2_df = st.session_state['raw2_df']
    target_col = st.session_state.get('selected_target', raw2_df.columns[0])
    if st.button("Train Model", key="btn_train_model"):
        try:
            X = raw2_df.drop(columns=[target_col])
            y = raw2_df[target_col]

            if y.dtype == float or y.nunique() > 10:
                st.warning("Target appears continuous. Attempting to binarize based on threshold 0.5.")
                y = (y >= 0.5).astype(int)

            vc = y.value_counts()
            valid_classes = vc[vc >= 2].index.tolist()
            if len(valid_classes) < 2:
                st.error("❌ Not enough classes with ≥2 samples for stratified split.")
                st.stop()

            mask = y.isin(valid_classes)
            X = X[mask]
            y = y[mask]

            smote = SMOTE(random_state=getattr(AppSettings, "RANDOM_STATE", 42))
            X_resampled, y_resampled = smote.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, stratify=y_resampled,
                random_state=getattr(AppSettings, "RANDOM_STATE", 42)
            )

            model = train_logistic_regression(X_train, y_train)
            y_probs = model.predict_proba(X_test)[:, 1]

            fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)
            precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)

            youden_j_scores = tpr - fpr
            optimal_threshold_j = thresholds_roc[np.argmax(youden_j_scores)]
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            optimal_threshold_f1 = thresholds_pr[np.argmax(f1_scores)]

            st.write(f"📌 Best Threshold (Youden's J): `{optimal_threshold_j:.2f}`")
            st.write(f"📌 Best Threshold (Max F1): `{optimal_threshold_f1:.2f}`")

            auto_choice = st.radio("Select auto-threshold strategy to apply:", options=["Manual", "YoudenJ", "MaxF1"], index=0)

            if auto_choice == "YoudenJ":
                threshold = optimal_threshold_j
            elif auto_choice == "MaxF1":
                threshold = optimal_threshold_f1
            else:
                threshold = st.slider("Select classification threshold", 0.0, 1.0, 0.5, 0.01)

            y_pred_thresh = (y_probs >= threshold).astype(int)

            acc = accuracy_score(y_test, y_pred_thresh)
            balanced_acc = balanced_accuracy_score(y_test, y_pred_thresh)
            f1 = f1_score(y_test, y_pred_thresh)

            st.success(f"✅ Training complete. Accuracy: {acc:.4f}")
            st.info(f"Balanced Accuracy: {balanced_acc:.4f} | F1 Score: {f1:.4f}")

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred_thresh))

            st.subheader("📉 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(1, 1))
            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_thresh, ax=ax)
            ax.set_title("Confusion Matrix", fontsize=7)
            ax.set_xlabel("Predicted Label", fontsize=5)
            ax.set_ylabel("True Label", fontsize=5)
            st.pyplot(fig)

            st.subheader("📈 ROC Curve")
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False Positive Rate", fontsize=5)
            ax.set_ylabel("True Positive Rate", fontsize=5)
            ax.set_title("ROC Curve", fontsize=7)
            ax.legend(loc="lower right", fontsize=5)
            st.pyplot(fig)

            st.subheader("📈 Precision-Recall Curve")
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot(recall, precision)
            ax.set_xlabel("Recall", fontsize=5)
            ax.set_ylabel("Precision", fontsize=5)
            ax.set_title("Precision-Recall Curve", fontsize=7)
            st.pyplot(fig)

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
