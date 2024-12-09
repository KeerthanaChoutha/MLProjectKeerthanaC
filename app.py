import streamlit as st
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# Extract and load the dataset
def load_data(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    return pd.read_csv("hate_crime.csv")

# App title
st.title("Hate Crime Hotspot Prediction")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (ZIP format):", type=["zip"])
if uploaded_file:
    with st.spinner("Loading and preprocessing the dataset..."):
        data = load_data(uploaded_file)

        # Sample the data to improve performance
        data = data.sample(n=5000, random_state=42)

    # Preview dataset
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Data cleaning options
    st.write("### Data Cleaning Options")
    if st.checkbox("Drop rows with missing values"):
        with st.spinner("Dropping missing values..."):
            data = data.dropna()
        st.write("Missing values dropped.")

    # Fill missing values with zero
    with st.spinner("Filling missing values..."):
        data = data.fillna(0)

    # Data exploration
    st.write("### Basic Statistics")
    st.write(data.describe())

    # Feature selection for modeling
    st.write("### Model Training")
    target = st.selectbox("Select Target Column:", options=data.columns)
    features = st.multiselect("Select Feature Columns (limit to 10):", options=[col for col in data.columns if col != target])

    if features and target:
        with st.spinner("Preparing data for training..."):
            X = data[features]
            y = data[target]

            # Encode target column if it is categorical
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Encode categorical features
            X = pd.get_dummies(X, drop_first=True)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Logistic Regression
        st.write("#### Logistic Regression")
        with st.spinner("Training Logistic Regression model..."):
            log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')
            try:
                log_reg.fit(X_train, y_train)
                st.write("Logistic Regression Performance:")
                st.text(classification_report(y_test, log_reg.predict(X_test)))
            except Exception as e:
                st.error(f"Logistic Regression failed: {e}")

        # Train Random Forest
        st.write("#### Random Forest Classifier")
        with st.spinner("Training Random Forest model..."):
            rf = RandomForestClassifier(n_jobs=-1)
            rf.fit(X_train, y_train)
            st.write("Random Forest Performance:")
            st.text(classification_report(y_test, rf.predict(X_test)))

        # Feature importance (SHAP)
        st.write("### Feature Importance (SHAP)")
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X.iloc[:1000])  # Limit to first 1000 rows
            plt.title("Feature Importance")
            shap.summary_plot(shap_values[1], X.iloc[:1000], show=False)
            st.pyplot(plt.gcf())
            plt.clf()

    # Geospatial clustering (Example with Plotly)
    if "state_name" in data.columns and "bias_desc" in data.columns:
        st.write("### Geospatial Insights")
        with st.spinner("Generating geospatial visualization..."):
            fig = px.scatter_geo(
                data,
                locations="state_name",
                locationmode="USA-states",
                color="bias_desc",
                scope="usa",
                title="Bias Descriptions by State",
            )
            st.plotly_chart(fig)
else:
    st.info("Upload a ZIP file containing your dataset to begin.")

