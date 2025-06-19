import streamlit as st
from src.config.settings import AppSettings
from src.utils.logging import setup_logging


logger = setup_logging()

st.set_page_config(
    page_title="Logistic Regression System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styles
st.markdown(
    """
    <style>
        body {
            background-color: #F1F0E8;
        }
        .stApp {
            background-color: #F1F0E8;
        }
        .stSidebar {
            background-color: #B3C8CF;
        }
        .stSidebar > div:first-child {
            background-color: #89A8B2;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 75vh;
            text-align: center;
        }
        .title-text {
            font-size: 3rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
        }
        .animated-subtitle {
            font-size: 1.8rem;
            font-weight: 500;
            color: #555;
            animation: fadeInSlide 2s ease-in-out forwards;
            opacity: 0;
        }
        @keyframes fadeInSlide {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0px);
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered Layout
st.markdown(
    """
    <div class="center-container">
        <div class="title-text">📊 Logistic Regression ML System</div>
        <div class="animated-subtitle">Welcome to the Logistic Regression Pipeline 🚀</div>
    </div>
    """,
    unsafe_allow_html=True
)
