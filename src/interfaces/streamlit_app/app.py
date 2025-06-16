import streamlit as st
from src.config.settings import AppSettings
from src.utils.logging import setup_logging
import os

logger = setup_logging()

st.set_page_config(
    page_title=AppSettings.STREAMLIT_APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(AppSettings.STREAMLIT_APP_TITLE)
st.sidebar.title("Navigation")

if st.session_state.get('current_page') is None:
    st.markdown(
        """
        Welcome to the **Logistic Regression ML Pipeline**!

        Use the sidebar to navigate through the different stages of the ML workflow:
        - **Upload Dataset**: Load your raw data.
        - **Preprocessing**: Clean and transform your data.
        - **Visualize**: Explore your raw and processed datasets.
        - **Inference**: Make predictions using the trained model.
        """
    )
    st.image(os.path.join(os.path.dirname(__file__), 'assets', 'ml_workflow.png')) # Placeholder image