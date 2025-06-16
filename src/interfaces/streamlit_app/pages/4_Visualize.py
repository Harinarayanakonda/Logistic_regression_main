import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.interfaces.streamlit_app.components.visualization import display_eda
from src.utils.logging import logger
from src.config.settings import AppSettings
import os

logger.info("Loading Visualize page.")

st.markdown("# 📊 Data Visualization & Comparison")

tab1, tab2, tab3 = st.tabs([
    "EDA1 - Raw Dataset",
    "EDA2 - Processed Dataset",
    "Compare Raw vs Processed"
])

with tab1:
    st.header("Exploratory Data Analysis (EDA1) – Raw Dataset")
    if 'raw1_df' in st.session_state and st.session_state['raw1_df'] is not None:
        raw1_df = st.session_state['raw1_df']
        try:
            display_eda(raw1_df, title_suffix="– Raw Dataset (raw1_df)", key_prefix="eda1")
        except Exception as e:
            st.error(f"EDA display failed: {e}")
            st.warning("If you see an expander error, please remove any st.expander usage from display_eda and its helpers.")
        if st.button("Refresh Raw EDA", key="refresh_eda1"):
            st.rerun()
    else:
        st.warning("No raw dataset (raw1_df) found. Please upload data on the 'Upload Dataset' page.")

with tab2:
    st.header("🔍 Exploratory Data Analysis – Processed Dataset (EDA2)")
    if 'raw2_df' in st.session_state and st.session_state['raw2_df'] is not None:
        raw2_df = st.session_state['raw2_df']
        try:
            display_eda(raw2_df, title_suffix="– Processed Dataset (raw2_df)", key_prefix="eda2")
        except Exception as e:
            st.error(f"EDA display failed: {e}")
            st.warning("If you see an expander error, please remove any st.expander usage from display_eda and its helpers.")
        if st.button("Refresh Processed EDA", key="refresh_eda2"):
            st.rerun()

        st.markdown("---")
        st.subheader("🤖 Train & Save Dummy Model")
        target_col = st.session_state.get('selected_target', raw2_df.columns[0])
        if st.button("Train & Save Dummy Model", key="btn_train_save"):
            try:
                X = raw2_df.drop(columns=[target_col])
                y = raw2_df[target_col]
                # Show class distribution
                st.write("Class distribution in target:")
                st.write(y.value_counts())
                # Remove classes with <2 samples
                vc = y.value_counts()
                valid_classes = vc[vc > 1].index
                X = X[y.isin(valid_classes)]
                y = y[y.isin(valid_classes)]
                if len(valid_classes) < 2:
                    st.error("Not enough classes with at least 2 samples for stratified split. Please check your data or preprocessing.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2,
                        random_state=getattr(AppSettings, "RANDOM_STATE", 42),
                        stratify=y
                    )
                    st.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
                    model = LogisticRegression(
                        random_state=getattr(AppSettings, "RANDOM_STATE", 42),
                        solver='liblinear'
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Accuracy on test set: {acc:.4f}")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                    # Save model to project directory
                    model_dir = os.path.join("src", "models")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, "dummy_logistic_model.pkl")
                    # Save both model and feature names
                    model_artifact = {
                        "model": model,
                        "feature_names": list(X.columns)
                    }
                    with open(model_path, "wb") as f:
                        pickle.dump(model_artifact, f)
                    st.success(f"Model and feature names saved to: {model_path}")
                    # Also provide download button
                    buffer = BytesIO()
                    pickle.dump(model_artifact, buffer)
                    buffer.seek(0)
                    st.download_button(
                        label="Download Trained Model (.pkl)",
                        data=buffer,
                        file_name="dummy_logistic_model.pkl",
                        mime="application/octet-stream"
                    )
            except Exception as e:
                st.error(f"🚫 Model training failed: {e}")

        if st.button("Back to Preprocessing", key="btn_back_preprocessing"):
            st.session_state['current_page'] = 'Preprocessing'
            st.switch_page("pages/2_Preprocessing.py")  # Change to "pages/3_Preprocessing.py" if needed
    else:
        st.warning("No processed dataset (raw2_df) found. Please run preprocessing on the 'Preprocessing' page.")

with tab3:
    st.header("🔄 Compare Raw vs Processed Data")
    if (
        'raw1_df' in st.session_state and st.session_state['raw1_df'] is not None and
        'raw2_df' in st.session_state and st.session_state['raw2_df'] is not None
    ):
        raw1_df = st.session_state['raw1_df']
        raw2_df = st.session_state['raw2_df']
        import matplotlib.pyplot as plt
        import seaborn as sns

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Dataset")
            st.dataframe(raw1_df.head())
            st.write(raw1_df.describe(include='all'))
            st.markdown("**Correlation Heatmap (Raw Data)**")
            num_cols = raw1_df.select_dtypes(include='number').columns
            if len(num_cols) > 1:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(raw1_df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Not enough numeric columns for heatmap.")
        with col2:
            st.subheader("Processed Dataset")
            st.dataframe(raw2_df.head())
            st.write(raw2_df.describe(include='all'))
            st.markdown("**Correlation Heatmap (Processed Data)**")
            num_cols = raw2_df.select_dtypes(include='number').columns
            if len(num_cols) > 1:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(raw2_df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Not enough numeric columns for heatmap.")
        if st.button("Refresh Comparison", key="refresh_compare"):
            st.rerun()
    else:
        st.warning("Both raw and processed datasets are required for comparison. Please upload and preprocess your data first.")