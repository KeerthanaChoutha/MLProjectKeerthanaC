import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# Helper function to load and extract ZIP file
def load_data(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()  # Extract all files in the current directory
    try:
        data = pd.read_csv("hate_crime.csv")  # Load the dataset
        st.write("Data loaded successfully. Preview:")
        st.dataframe(data.head())
        return data
    except FileNotFoundError:
        st.error("The file 'hate_crime.csv' was not found in the ZIP. Ensure the ZIP contains this file.")
        return None

# App title
st.title("Hate Crime Hotspot Prediction")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (ZIP format containing 'hate_crime.csv'):", type=["zip"])

if uploaded_file:
    # Load the data
    data = load_data(uploaded_file)

    if data is not None:
        # Handle missing values
        st.write("### Missing Value Summary")
        st.write(data.isnull().sum())

        # Cleaning options
        st.write("### Data Cleaning Options")
        if st.checkbox("Drop rows with missing values"):
            data = data.dropna()
            st.write("Missing values dropped.")
            st.dataframe(data.head())

        # Data exploration
        st.write("### Basic Statistics")
        st.write(data.describe())

        # Feature and target selection for modeling
        st.write("### Model Training")
        target = st.selectbox("Select Target Column:", options=data.columns)
        features = st.multiselect("Select Feature Columns:", options=[col for col in data.columns if col != target])

        if features and target:
            X = data[features]
            y = data[target]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train Logistic Regression
            st.write("#### Logistic Regression")
            try:
                log_reg = LogisticRegression(max_iter=1000)
                log_reg.fit(X_train, y_train)
                st.write("Logistic Regression Performance:")
                st.text(classification_report(y_test, log_reg.predict(X_test)))
            except Exception as e:
                st.error(f"Error in Logistic Regression: {e}")

            # Train Random Forest
            st.write("#### Random Forest Classifier")
            try:
                rf = RandomForestClassifier()
                rf.fit(X_train, y_train)
                st.write("Random Forest Performance:")
                st.text(classification_report(y_test, rf.predict(X_test)))
            except Exception as e:
                st.error(f"Error in Random Forest: {e}")

            # Feature importance (SHAP)
            st.write("### Feature Importance (SHAP)")
            try:
                explainer = shap.TreeExplainer(rf)
                sample_data = X.sample(100, random_state=42)  # Sample for SHAP to avoid memory issues
                shap_values = explainer.shap_values(sample_data)
                plt.title("Feature Importance")
                shap.summary_plot(shap_values[1], sample_data, show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception as e:
                st.error(f"Error in SHAP Plot: {e}")

        # Geospatial clustering example with Plotly
        if "state_name" in data.columns and "bias_desc" in data.columns:
            st.write("### Geospatial Insights")
            try:
                fig = px.scatter_geo(
                    data,
                    locations="state_name",
                    locationmode="USA-states",
                    color="bias_desc",
                    scope="usa",
                    title="Bias Descriptions by State",
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error in Geospatial Visualization: {e}")
else:
    st.info("Upload a ZIP file containing your dataset to begin.")
