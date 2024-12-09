import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# Extract and load the dataset
def load_data(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("extracted_data")
        extracted_files = os.listdir("extracted_data")
        for file in extracted_files:
            if file.endswith(".csv"):
                return pd.read_csv(os.path.join("extracted_data", file))
        st.error("No CSV file found in the ZIP archive.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return None

# App title
st.title("Hate Crime Hotspot Prediction")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (ZIP format):", type=["zip"])
if uploaded_file:
    st.write("Processing the uploaded file...")
    data = load_data(uploaded_file)
    if data is not None:
        try:
            # Preview dataset
            st.write("### Dataset Preview")
            st.dataframe(data.head())

            # Data cleaning options
            st.write("### Data Cleaning Options")
            if st.checkbox("Drop rows with missing values"):
                data = data.dropna()
                st.write("Missing values dropped.")
            
            # Data exploration
            st.write("### Basic Statistics")
            st.write(data.describe())

            # Feature selection for modeling
            st.write("### Model Training")
            target = st.selectbox("Select Target Column:", options=data.columns)
            features = st.multiselect("Select Feature Columns:", options=[col for col in data.columns if col != target])

            if features and target:
                X = data[features]
                y = data[target]

                # Encode categorical variables
                le = LabelEncoder()
                if y.dtype == 'object':
                    y = le.fit_transform(y)
                X = pd.get_dummies(X, drop_first=True)

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
                    st.error(f"Logistic Regression Error: {e}")

                # Train Random Forest
                st.write("#### Random Forest Classifier")
                rf = RandomForestClassifier()
                rf.fit(X_train, y_train)
                st.write("Random Forest Performance:")
                st.text(classification_report(y_test, rf.predict(X_test)))

                # Feature importance (SHAP)
                st.write("### Feature Importance (SHAP)")
                try:
                    explainer = shap.TreeExplainer(rf)
                    shap_values = explainer.shap_values(X_test)
                    plt.title("Feature Importance")
                    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.error(f"SHAP Error: {e}")

            # Geospatial clustering (Example with Plotly)
            if "state_name" in data.columns and "bias_desc" in data.columns:
                st.write("### Geospatial Insights")
                try:
                    fig = px.scatter_geo(
                        data,
                        locations="state_abbr",
                        locationmode="USA-states",
                        color="bias_desc",
                        scope="usa",
                        title="Bias Descriptions by State",
                        hover_name="state_name"
                    )
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Geospatial Visualization Error: {e}")

        except Exception as main_e:
            st.error(f"Unexpected error: {main_e}")
else:
    st.info("Upload a ZIP file containing your dataset to begin.")
