import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime  # Added this import

# Apply page config
st.set_page_config(page_title="EDA Dashboard", layout="wide")

# Inject updated custom styles with the new palette
st.markdown("""
    <style>
    .stApp {
        background-color: #EAE4D5;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #332D56;
    }
    .section {
        background-color: #4E6688;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .inner-section {
        background-color: #EAE4D5;
        padding: 1rem;
        border-radius: 10px;
        color: #332D56;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #332D56;
    }
    </style>
""", unsafe_allow_html=True)

def load_latest_raw_dataset():
    """Load the most recent raw dataset from the data directory"""
    raw_dir = Path("data/01_raw")
    if not raw_dir.exists():
        return None
    
    # Get all CSV files sorted by modification time (newest first)
    csv_files = sorted(raw_dir.glob("*.csv"), key=os.path.getmtime, reverse=True)
    
    if not csv_files:
        return None
    
    try:
        latest_file = csv_files[0]
        return pd.read_csv(latest_file), latest_file
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

st.markdown("# 📊 EDA: Univariate & Bivariate Analysis")

# Load DataFrame from saved raw data
result = load_latest_raw_dataset()
if result is None:
    st.warning("⚠️ No raw dataset found. Please upload a dataset first.")
    st.stop()

df, data_path = result
st.info(f"Analyzing dataset: {data_path.name}")

# Univariate Analysis
st.subheader("🔍 Univariate Analysis")
selected_col = st.selectbox("Select a column", df.columns)

if pd.api.types.is_numeric_dtype(df[selected_col]):
    st.markdown('<div class="inner-section">', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="#71C0BB")
    ax.set_facecolor("#EAE4D5")
    ax.set_title(f"Distribution of {selected_col}")
    st.pyplot(fig)
    st.write("📊 Summary Statistics")
    st.dataframe(df[selected_col].describe())
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="inner-section">', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    df[selected_col].value_counts().plot(kind='bar', ax=ax, color="#71C0BB")
    ax.set_facecolor("#EAE4D5")
    ax.set_title(f"Value Counts for {selected_col}")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("📋 Value Counts")
    st.dataframe(df[selected_col].value_counts())
    st.markdown('</div>', unsafe_allow_html=True)

# Bivariate Analysis
st.subheader("🔗 Bivariate Analysis")
col1 = st.selectbox("X-axis", df.columns, key="biv_x")
col2 = st.selectbox("Y-axis", df.columns, key="biv_y")

if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
    st.markdown('<div class="inner-section">', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[col1], y=df[col2], ax=ax, color="#71C0BB")
    ax.set_facecolor("#EAE4D5")
    ax.set_title(f"{col1} vs {col2}")
    st.pyplot(fig)
    st.write("📈 Correlation:", round(df[[col1, col2]].corr().iloc[0, 1], 3))
    st.markdown('</div>', unsafe_allow_html=True)
elif pd.api.types.is_numeric_dtype(df[col1]) != pd.api.types.is_numeric_dtype(df[col2]):
    st.markdown('<div class="inner-section">', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    if pd.api.types.is_numeric_dtype(df[col1]):
        sns.boxplot(x=df[col1], y=df[col2], ax=ax, palette=["#71C0BB"])
    else:
        sns.boxplot(x=df[col2], y=df[col1], ax=ax, palette=["#71C0BB"])
    ax.set_facecolor("#EAE4D5")
    ax.set_title(f"Distribution of {col2} by {col1}")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="inner-section">', unsafe_allow_html=True)
    crosstab = pd.crosstab(df[col1], df[col2])
    fig, ax = plt.subplots()
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_facecolor("#EAE4D5")
    ax.set_title(f"Cross-tabulation: {col1} vs {col2}")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Dataset Info Section
with st.expander("🔍 Dataset Information"):
    st.write(f"**Dataset:** {data_path.name}")
    st.write(f"**Last Modified:** {datetime.fromtimestamp(data_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Columns:**", df.columns.tolist())