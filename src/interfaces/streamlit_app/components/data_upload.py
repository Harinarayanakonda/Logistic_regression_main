import streamlit as st
from io import StringIO, BytesIO
from src.core.data_loader import DataLoader
from src.utils.logging import logger

def data_upload_interface():
    st.header("Upload Dataset")
    st.write("Select your dataset file and format to begin the preprocessing workflow.")

    col1, col2 = st.columns([1, 2])

    with col1:
        file_format = st.selectbox(
            "Select File Format",
            ("csv", "txt", "xlsx", "xls", "xlsm", "json")
        )
    
    with col2:
        uploaded_file = st.file_uploader(
            "Drag and drop your file here or click to browse",
            type=["csv", "txt", "xlsx", "xls", "xlsm", "json"],
            accept_multiple_files=False,
            key="data_uploader"
        )

    submit_button = st.button("Submit", key="submit_upload")
    cancel_button = st.button("Cancel", key="cancel_upload")

    if cancel_button:
        uploaded_file = None # Clear uploaded file
        if 'raw_df' in st.session_state:
            del st.session_state['raw_df']
        if 'raw1_df' in st.session_state:
            del st.session_state['raw1_df']
        st.info("Upload cancelled. Data cleared from session.")
        st.rerun()

    if submit_button and uploaded_file is not None:
        try:
            data_loader = DataLoader()
            if file_format in ['csv', 'txt']:
                raw_df = data_loader.load_data(StringIO(uploaded_file.getvalue().decode('utf-8')))
            elif file_format in ['xlsx', 'xls', 'xlsm']:
                raw_df = data_loader.load_data(BytesIO(uploaded_file.getvalue()))
            elif file_format == 'json':
                raw_df = data_loader.load_data(StringIO(uploaded_file.getvalue().decode('utf-8')))
            else:
                st.error("Unsupported file format selected.")
                raw_df = None

            if raw_df is not None:
                st.session_state['raw_df'] = raw_df
                st.session_state['raw1_df'] = raw_df.copy()
                st.success("✅ Dataset loaded successfully!")
                st.write(f"**Dataset Dimensions:** {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
                st.subheader("First 5 records:")
                st.dataframe(raw_df.head())
            else:
                st.error("Failed to load data. Please check the file and format.")

        except Exception as e:
            logger.error(f"Error during file upload or loading: {e}")
            st.error(f"An error occurred: {e}. Please try again.")
    elif submit_button and uploaded_file is None:
        st.warning("Please upload a file before clicking Submit.")