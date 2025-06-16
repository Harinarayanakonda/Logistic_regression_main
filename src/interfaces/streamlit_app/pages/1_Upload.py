import streamlit as st
from src.interfaces.streamlit_app.components.data_upload import data_upload_interface
from src.utils.logging import logger

logger.info("Loading Upload page.")

st.markdown("# Upload Dataset")
data_upload_interface()

# Persist raw1_df across pages using session_state
if 'raw_df' in st.session_state and st.session_state['raw_df'] is not None:
    st.session_state['raw1_df'] = st.session_state['raw_df']  # Ensure consistency
    st.sidebar.success("Dataset loaded!")
    st.markdown("## Preview of Uploaded Data")
    st.dataframe(st.session_state['raw_df'].head(), use_container_width=True)
    st.write("**Columns in uploaded data:**", st.session_state['raw_df'].columns.tolist())
else:
    st.sidebar.warning("No dataset loaded.")
    st.info("Please upload a dataset to continue.")