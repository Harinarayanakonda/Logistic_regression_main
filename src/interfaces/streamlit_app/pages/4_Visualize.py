import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shutil
from io import BytesIO
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, precision_recall_curve, f1_score, roc_curve, auc
from src.interfaces.streamlit_app.components.visualization import display_eda
from src.utils.logging import logger
from src.config.settings import AppSettings
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
    processed_df = st.session_state['raw2_df']
    excluded_cols = st.session_state.get('excluded_columns', [])
    if excluded_cols:
        processed_df = processed_df.drop(columns=excluded_cols, errors='ignore')
    display_eda(processed_df, title_suffix="(Processed)", key_prefix="eda_processed")
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

# Fixed model directory
FIXED_MODEL_DIR = "src/models/Trained_model"
os.makedirs(FIXED_MODEL_DIR, exist_ok=True)
st.session_state["session_model_dir"] = FIXED_MODEL_DIR

# --------------------------- TRAINING SECTION ---------------------------
st.subheader("🤔 Train Model on Preprocessed Data")
if 'raw2_df' in st.session_state and st.session_state['raw2_df'] is not None:
    raw2_df = st.session_state['raw2_df']
    excluded_cols = st.session_state.get('excluded_columns', [])
    raw2_df = raw2_df.drop(columns=excluded_cols, errors='ignore')
    st.session_state['raw2_df'] = raw2_df

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
            threshold = optimal_threshold_j if auto_choice == "YoudenJ" else optimal_threshold_f1 if auto_choice == "MaxF1" else st.slider("Select classification threshold", 0.0, 1.0, 0.5, 0.01)

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
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred_thresh, ax=ax)
            st.pyplot(fig)

            st.subheader("📈 ROC Curve")
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.legend(loc="lower right", fontsize=5)
            st.pyplot(fig)

            st.subheader("📊 Precision-Recall Curve")
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot(recall, precision)
            st.pyplot(fig)

            # ✅ Save model and training info
            st.session_state["trained_model"] = model
            st.session_state["model_input_sample"] = X_train.head(1)
            st.session_state["model_accuracy"] = acc
            st.session_state["X_train"] = X_train
            st.session_state["final_preprocessor"] = None  # Update if you have a pipeline

        except Exception as e:
            st.error(f"Model training failed: {e}")
# -------------------- SAVE TRAINED MODEL ONLY --------------------
def save_model_artifacts(model):
    artifacts_dir = "src/models/Trained_model"
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(
        artifacts_dir,
        f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path

# ------------------------ SAVE TRIGGER ------------------------
# ------------------------ SAVE TRIGGER ------------------------
if "trained_model" in st.session_state:
    st.subheader("📅 Save Trained Model")

    if st.button("Save Model", key="btn_save_model"):
        try:
            model = st.session_state["trained_model"]
            X_train = st.session_state.get("X_train")

            if X_train is None:
                st.warning("Missing training data. Cannot save.")
                st.stop()

            st.warning("No preprocessor found. Saving model only.")

            model_path = save_model_artifacts(model)

            st.success("✅ Model saved successfully.")
            st.json({
                "model_path": model_path
            })

            # Clear session state
            del st.session_state["trained_model"]
            st.session_state.pop("model_input_sample", None)
            st.session_state.pop("model_accuracy", None)

        except Exception as e:
            st.error(f"❌ Saving model failed: {e}")

# ------------------------ CLEANUP ------------------------
if st.button("End Session and Clean Model Folder"):
    try:
        session_dir = st.session_state.get("session_model_dir", "")
        shutil.rmtree(session_dir, ignore_errors=True)
        st.success("🗑️ Session model directory cleaned up.")
        st.session_state.pop("session_model_dir", None)
    except Exception as e:
        st.error(f"Cleanup failed: {e}")
