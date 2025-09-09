import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lazypredict.Supervised import LazyClassifier, LazyRegressor

st.set_page_config(page_title="Auto Data Analyzer + AutoML", layout="wide")

st.title("📊 Automated Data Analyzer + AutoML (LazyPredict)")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Auto-detect separator
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        st.error("Error reading CSV file")
        st.stop()

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📋 Data Overview")
    st.write("Shape:", df.shape)
    st.write("Column Types:")
    st.write(df.dtypes)

    # Missing values
    st.subheader("❌ Missing Values")
    st.write(df.isnull().sum())

    # Descriptive stats
    st.subheader("📈 Descriptive Statistics")
    st.write(df.describe(include="all"))

    # Detect column types
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object','category']).columns

    st.subheader("📊 Numerical Analysis")
    for col in num_cols:
        st.write(f"Distribution of **{col}**")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("🗂️ Categorical Analysis")
    for col in cat_cols:
        st.write(f"Count plot of **{col}**")
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], order=df[col].value_counts().index[:10], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if len(num_cols) > 1:
        st.subheader("📊 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ----------------- AutoML Section -----------------
    st.subheader("🤖 AutoML Model Comparison (LazyPredict)")

    target = st.selectbox("Select the target column", df.columns)

    if target:
        X = df.drop(columns=[target]).select_dtypes(include=['int64','float64'])
        y = df[target]

        if X.empty:
            st.warning("No numeric features available for modeling.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Decide classification vs regression
            if y.nunique() <= 10 and y.dtype != "float64":
                st.write("🧮 Detected **Classification** problem")
                clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
                models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                st.write("### Model Comparison Results")
                st.dataframe(models)

                # Train a simple RandomForestClassifier for predictions
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                results = pd.DataFrame({"Actual": y_test, "Predicted": preds})
                st.write("### Predictions on Test Data")
                st.dataframe(results.head(20))

            else:
                st.write("📈 Detected **Regression** problem")
                reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
                models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                st.write("### Model Comparison Results")
                st.dataframe(models)

                # Train a simple RandomForestRegressor for predictions
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                results = pd.DataFrame({"Actual": y_test, "Predicted": preds})
                st.write("### Predictions on Test Data")
                st.dataframe(results.head(20))
