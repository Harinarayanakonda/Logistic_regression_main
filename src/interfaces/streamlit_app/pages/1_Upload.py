import streamlit as st
import pandas as pd
import os
import atexit
from datetime import datetime
from pathlib import Path
from src.interfaces.streamlit_app.components.data_upload import data_upload_interface
from src.utils.logging import logger

logger.info("Loading Upload page.")

# Cleanup on exit
def cleanup_saved_data():
    raw_dir = Path("data/01_raw")
    if raw_dir.exists():
        for file in raw_dir.glob("*.csv"):
            try:
                os.remove(file)
                logger.info(f"Removed temporary file: {file}")
            except Exception as e:
                logger.error(f"Failed to remove {file}: {e}")

atexit.register(cleanup_saved_data)

# Styling
st.set_page_config(page_title="Upload Dataset", layout="wide")
st.markdown("""
    <style>
        .upload-section {
            text-align: center;
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .upload-header {
            font-size: 32px;
            font-weight: bold;
            color: #2C3E50;
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
""", unsafe_allow_html=True)

def ensure_raw_directory():
    raw_dir = Path("data/01_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def save_raw_dataset(df, filename=None):
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

# Upload Section Header
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="upload-header">📂 Upload Your Dataset</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-subtext">Drag and drop your file here or click to browse</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Call upload interface (DO NOT store in a variable)
data_upload_interface()

# Post-upload block
if 'raw_df' in st.session_state and st.session_state['raw_df'] is not None:
    raw_df = st.session_state['raw_df']

    # Column Exclusion UI
    with st.container():
        st.markdown('<div class="uploaded-df">', unsafe_allow_html=True)
        st.write("**Columns Detected:**")
        st.table(pd.DataFrame(raw_df.columns, columns=["Column Name"]))

        st.subheader("🛑 Select Columns to Exclude from EDA and Preprocessing")
        exclude_cols = st.multiselect(
            "Choose the columns you want to skip:",
            options=raw_df.columns.tolist(),
            default=st.session_state.get('exclude_columns', []),
            key='column_exclusion_selector'
        )

        # Update session and filter data
        st.session_state['exclude_columns'] = exclude_cols
        filtered_df = raw_df.drop(columns=exclude_cols) if exclude_cols else raw_df.copy()
        st.session_state['raw1_df'] = filtered_df

        # Feedback
        st.info(f"Excluded Columns: {', '.join(exclude_cols) if exclude_cols else 'None'}")
        st.write(f"**Remaining Columns:** {len(filtered_df.columns)}")
        st.write("**Filtered Dataset Preview:**")
        st.dataframe(filtered_df.head())
        st.markdown('</div>', unsafe_allow_html=True)

    # Save Options
    with st.expander("💾 Save Options", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save Original Dataset"):
                try:
                    saved_path = save_raw_dataset(raw_df)
                    st.success(f"✅ Full dataset saved to: {saved_path}")
                except Exception as e:
                    st.error(f"Save failed: {str(e)}")

        with col2:
            if st.button("Save Filtered Dataset"):
                try:
                    saved_path = save_raw_dataset(
                        filtered_df,
                        f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                    st.success(f"✅ Filtered dataset saved to: {saved_path}")
                except Exception as e:
                    st.error(f"Save failed: {str(e)}")

        custom_name = st.text_input(
            "Custom filename (without extension)",
            value=f"dataset_{datetime.now().strftime('%Y%m%d')}"
        )
        if st.button("💾 Save with Custom Name"):
            try:
                save_path = save_raw_dataset(
                    filtered_df,
                    f"{custom_name}.csv"
                )
                st.success(f"✅ Dataset saved as: {save_path}")
            except Exception as e:
                st.error(f"Save failed: {str(e)}")

    st.sidebar.success("✅ Dataset ready for analysis!")
    st.sidebar.info(f"📊 Columns available: {len(filtered_df.columns)}")

else:
    st.sidebar.warning("⚠️ No dataset found.")
    st.info("Please upload a dataset to begin analysis.")
