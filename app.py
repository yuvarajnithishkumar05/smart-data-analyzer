import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Smart Data Analyzer")

st.write("Upload your dataset and get automatic analysis, cleaning and visualization")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.header("Dataset Preview")
    st.write(df.head())
    st.write("Shape of dataset:", df.shape)

    st.header("Dataset Report")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.write("Duplicate rows:", df.duplicated().sum())
    st.write("Total missing values:", df.isnull().sum().sum())

    st.header("Data Types")
    st.write(df.dtypes)

    st.header("Missing Values Analysis")

    missing = df.isnull().sum()
    st.write(missing)
    st.bar_chart(missing)

    st.header("Data Cleaning")

    if st.checkbox("Remove duplicates"):
        df = df.drop_duplicates()
        st.success("Duplicates removed")

    if st.checkbox("Fill missing values with mean (numerical columns)"):
        df = df.fillna(df.mean(numeric_only=True))
        st.success("Missing values filled")

    st.write(df)

    st.header("Column Classification")

    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    st.write("Numerical Columns:", num_cols)
    st.write("Categorical Columns:", cat_cols)

    st.header("Summary Statistics")
    st.write(df.describe())

    st.header("Data Visualization")

    column = st.selectbox("Select column for visualization", df.columns)

    chart = st.selectbox(
        "Select chart type",
        ["Line Chart","Bar Chart","Histogram"]
    )

    if chart == "Line Chart":
        st.line_chart(df[column])

    elif chart == "Bar Chart":
        st.bar_chart(df[column])

    elif chart == "Histogram":

        fig, ax = plt.subplots()
        ax.hist(df[column])
        st.pyplot(fig)

    st.header("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(fig)

    st.header("Outlier Detection")

    if len(num_cols) > 0:

        col = st.selectbox("Select column for outlier detection", num_cols)

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]

        st.write("Number of outliers:", len(outliers))

    st.header("Data Filtering")

    filter_column = st.selectbox("Select column to filter", df.columns)

    value = st.text_input("Enter value to filter")

    if value:
        filtered_df = df[df[filter_column].astype(str) == value]
        st.write(filtered_df)

    st.header("Download Cleaned Dataset")

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "cleaned_data.csv",
        "text/csv"
    )