import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("# 📊 EDA 1: Univariate & Bivariate Analysis")

# Load the DataFrame
if 'raw_df' in st.session_state and st.session_state['raw_df'] is not None:
    df = st.session_state['raw_df']
else:
    st.warning("Please upload a dataset first.")
    st.stop()

st.subheader("Univariate Analysis")
selected_col = st.selectbox("Select a column for univariate analysis", df.columns)

if pd.api.types.is_numeric_dtype(df[selected_col]):
    st.write("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
    st.pyplot(fig)
    st.write("Summary Statistics")
    st.write(df[selected_col].describe())
else:
    st.write("Bar Plot")
    fig, ax = plt.subplots()
    df[selected_col].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    st.write("Value Counts")
    st.write(df[selected_col].value_counts())

st.markdown("---")
st.subheader("Bivariate Analysis")
col1 = st.selectbox("Select X (feature)", df.columns, key="biv_x")
col2 = st.selectbox("Select Y (feature)", df.columns, key="biv_y")

if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
    st.write("Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
    st.pyplot(fig)
    st.write("Correlation:", df[[col1, col2]].corr().iloc[0,1])
elif pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_numeric_dtype(df[col2]):
    st.write("Box Plot")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col2], y=df[col1], ax=ax)
    st.pyplot(fig)
elif not pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
    st.write("Box Plot")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col1], y=df[col2], ax=ax)
    st.pyplot(fig)
else:
    st.write("Heatmap of Counts")
    crosstab = pd.crosstab(df[col1], df[col2])
    fig, ax = plt.subplots()
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)