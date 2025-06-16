import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_numeric_and_categorical_features(df):
    numeric_features = df.select_dtypes(include='number').columns.tolist()
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    return numeric_features, categorical_features

def plot_univariate_numeric(df: pd.DataFrame, numeric_cols: list, key_prefix=""):
    st.subheader("Univariate Analysis: Numeric Features")
    if not numeric_cols:
        st.info("No numeric features available.")
        return
    selectbox_key = f"{key_prefix}_select_num_eda"
    selected_num_col = st.selectbox("Select a Numeric Feature", numeric_cols, key=selectbox_key)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Distribution of {selected_num_col}**")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num_col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.write(f"**Boxplot of {selected_num_col}**")
        fig, ax = plt.subplots()
        sns.boxplot(y=df[selected_num_col].dropna(), ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    st.subheader(f"Descriptive Statistics for {selected_num_col}")
    st.dataframe(df[selected_num_col].describe())

def plot_univariate_categorical(df: pd.DataFrame, categorical_cols: list, key_prefix=""):
    st.subheader("Univariate Analysis: Categorical Features")
    if not categorical_cols:
        st.info("No categorical features available.")
        return
    selectbox_key = f"{key_prefix}_select_cat_eda"
    selected_cat_col = st.selectbox("Select a Categorical Feature", categorical_cols, key=selectbox_key)
    st.write(f"**Value Counts for {selected_cat_col}**")
    fig, ax = plt.subplots()
    df[selected_cat_col].value_counts().plot(kind='bar', ax=ax)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.close(fig)
    st.subheader(f"Value Counts and Percentages for {selected_cat_col}")
    value_counts = df[selected_cat_col].value_counts()
    value_percentages = df[selected_cat_col].value_counts(normalize=True) * 100
    counts_df = pd.DataFrame({'Count': value_counts, 'Percentage': value_percentages.round(2)})
    st.dataframe(counts_df)

def plot_bivariate_analysis(df: pd.DataFrame, numeric_cols: list, key_prefix=""):
    st.subheader("Bivariate Analysis")
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns for bivariate analysis.")
        return
    bivariate_option = st.selectbox(
        "Choose Bivariate Plot Type",
        ["Correlation Heatmap", "Pair Plot (Sample)", "Scatter Plot"],
        key=f"{key_prefix}_bivariate_plot_type"
    )

    if bivariate_option == "Correlation Heatmap":
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    elif bivariate_option == "Pair Plot (Sample)":
        st.write("**Pair Plot (Sample of 1000 rows for performance)**")
        sample_df = df[numeric_cols].sample(min(1000, len(df)), random_state=42)
        fig = sns.pairplot(sample_df.dropna())
        st.pyplot(fig)
        plt.close(fig)

    elif bivariate_option == "Scatter Plot":
        col_x = st.selectbox("Select X-axis for Scatter Plot", numeric_cols, key=f"{key_prefix}_scatter_x")
        col_y = st.selectbox("Select Y-axis for Scatter Plot", numeric_cols, key=f"{key_prefix}_scatter_y")
        if col_x and col_y:
            st.write(f"**Scatter Plot: {col_x} vs {col_y}**")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

def display_eda(df: pd.DataFrame, title_suffix: str = "", key_prefix: str = ""):
    st.subheader(f"Exploratory Data Analysis {title_suffix}")
    if df.empty:
        st.warning("DataFrame is empty. Cannot perform EDA.")
        return

    numeric_features, categorical_features = get_numeric_and_categorical_features(df)
    st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.markdown(f"**Numeric Features ({len(numeric_features)}):** {', '.join(numeric_features) if numeric_features else 'None'}")
    st.markdown(f"**Categorical Features ({len(categorical_features)}):** {', '.join(categorical_features) if categorical_features else 'None'}")
    st.markdown("---")

    if numeric_features:
        plot_univariate_numeric(df, numeric_features, key_prefix=key_prefix)
    else:
        st.info("No numeric features for univariate analysis.")

    st.markdown("---")

    if categorical_features:
        plot_univariate_categorical(df, categorical_features, key_prefix=key_prefix)
    else:
        st.info("No categorical features for univariate analysis.")

    st.markdown("---")

    if numeric_features and len(numeric_features) > 1:
        plot_bivariate_analysis(df, numeric_features, key_prefix=key_prefix)
    else:
        st.info("Need at least two numeric features for bivariate analysis.")