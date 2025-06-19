import streamlit as st
import pandas as pd
import os
import atexit
from datetime import datetime
from pathlib import Path
from src.interfaces.streamlit_app.components.data_upload import data_upload_interface
from src.utils.logging import logger

logger.info("Loading Upload page.")

# Cleanup function to remove saved data when app closes
def cleanup_saved_data():
    """Remove all saved raw datasets when app exits"""
    raw_dir = Path("data/01_raw")
    if raw_dir.exists():
        for file in raw_dir.glob("*.csv"):
            try:
                os.remove(file)
                logger.info(f"Removed temporary file: {file}")
            except Exception as e:
                logger.error(f"Failed to remove {file}: {e}")

# Register cleanup function
atexit.register(cleanup_saved_data)

# Page Styling
st.set_page_config(page_title="Upload Dataset", layout="wide")

st.markdown(
    """
    <style>
        body {
            background-color: #F1EFEC;
        }
        .stApp {
            background-color: #F1EFEC;
        }
        .stSidebar {
            background-color: #D4C9BE;
        }
        .stSidebar > div:first-child {
            background-color: #123458;
        }
        h1, h2, h3, .stMarkdown h1 {
            color: #123458;
        }
        .stButton > button {
            background-color: #D4C9BE;
            color: #030303;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #123458;
            color: #ffffff;
            transform: scale(1.02);
        }
        .upload-section {
            text-align: center;
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .upload-header {
            font-size: 32px;
            font-weight: bold;
            color: #123458;
        }
        .upload-subtext {
            font-size: 18px;
            color: #555;
            margin-top: 8px;
        }
        .uploaded-df {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
def ensure_raw_directory():
    """Ensure the raw data directory exists"""
    raw_dir = Path("data/01_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def save_raw_dataset(df, filename=None):
    """Save raw dataset with timestamp"""
    raw_dir = ensure_raw_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename or f"raw_data_{timestamp}.csv"
    save_path = raw_dir / filename
    try:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved raw data to {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save raw data: {e}")
        raise

# Custom Upload Heading UI
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="upload-header">📂 Upload Your Dataset</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-subtext">Drag and drop your file here or click to browse</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Upload Interface
data_upload_interface()

# After Upload Processing
if 'raw_df' in st.session_state and st.session_state['raw_df'] is not None:
    # Save the raw dataset
    try:
        saved_path = save_raw_dataset(st.session_state['raw_df'])
        st.session_state['raw_data_path'] = str(saved_path)
        st.sidebar.success(f"✅ Dataset saved to: {saved_path}")
        st.sidebar.info("ℹ️ All saved datasets will be automatically removed when you close the app")
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to save raw data: {str(e)}")
    
    # Maintain existing functionality
    st.session_state['raw1_df'] = st.session_state['raw_df']
    
    # Dataset Preview
    with st.container():
        st.markdown('<div class="uploaded-df">', unsafe_allow_html=True)
        st.write("**Columns Detected:**")
        st.table(pd.DataFrame(st.session_state['raw_df'].columns, columns=["Column Name"]))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Optional Save Options
        with st.expander("🔧 Advanced Save Options", expanded=False):
            custom_name = st.text_input(
                "Custom filename (without extension)", 
                value=f"dataset_{datetime.now().strftime('%Y%m%d')}"
            )
            if st.button("💾 Save with custom name"):
                try:
                    custom_path = save_raw_dataset(
                        st.session_state['raw_df'],
                        f"{custom_name}.csv"
                    )
                    st.success(f"Dataset saved as: {custom_path}")
                except Exception as e:
                    st.error(f"Save failed: {str(e)}")
else:
    st.sidebar.warning("⚠️ No dataset found.")
    st.info("Please upload a dataset to begin analysis.")