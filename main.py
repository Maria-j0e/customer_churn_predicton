import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sklearn.compose._column_transformer as ct


# ==========================
# Fix for pickle load error
# ==========================
class _RemainderColsList(list):
    pass


ct._RemainderColsList = _RemainderColsList

# ==========================
# Title
# ==========================
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn or stay, based on their details.")

# ==========================
# Load the trained model (deployment-safe)
# ==========================
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "model" / "customer_churn_pipeline.pkl"  # Ensure lowercase 'model' folder

if not model_path.exists():
    st.error(f"Model file not found at {model_path}")
    st.stop()  # Stop the app if model not found

churn_predictor = joblib.load(model_path)

# ==========================
# User Inputs
# ==========================
st.subheader("Personal Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    SeniorCitizen = st.radio("Senior Citizen", ["Yes", "No"])
    Partner = st.radio("Partner", ["Yes", "No"])
    Dependents = st.radio("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 100, 0)

with col2:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.radio("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.slider("Monthly Charges ($)", 0, 150, 50)
    TotalCharges = st.slider("Total Charges ($)", 0, 10000, 1000)

st.subheader("Additional Information")

col3, col4 = st.columns(2)

with col3:
    PhoneService = st.radio("Phone Service", ["Yes", "No"], index=1)  # Default = No
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=1)  # Default = No
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=2)  # Default = No

    # Show/hide based on Internet Service
    if InternetService != "No":
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"], index=1)  # Default = No
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"], index=1)  # Default = No
    else:
        OnlineSecurity = "No internet service"
        OnlineBackup = "No internet service"

with col4:
    if InternetService != "No":
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"], index=1)  # Default = No
        TechSupport = st.selectbox("Tech Support", ["Yes", "No"], index=1)  # Default = No
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"], index=1)  # Default = No
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"], index=1)  # Default = No
    else:
        DeviceProtection = "No internet service"
        TechSupport = "No internet service"
        StreamingTV = "No internet service"
        StreamingMovies = "No internet service"


# Convert SeniorCitizen Yes/No → 1/0
SeniorCitizen_val = 1 if SeniorCitizen == "Yes" else 0

# ==========================
# Prepare DataFrame for Prediction
# ==========================
user_input = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen_val,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

st.write("Customer Details Summary")
st.dataframe(user_input)

# ==========================
# Prediction
# ==========================
if st.button("Predict Churn"):
    try:
        prediction = churn_predictor.predict(user_input)[0]
        prediction_proba = churn_predictor.predict_proba(user_input)[0]

        churn_probability = prediction_proba[1] * 100  # Probability of churning
        stay_probability = prediction_proba[0] * 100  # Probability of staying

        # Display prediction result
        if prediction == 1:
            st.error(f"The customer is likely to Churn ({churn_probability:.2f}% probability)")
        else:
            st.success(f"The customer is likely to Stay ({stay_probability:.2f}% probability)")

        # Display probability breakdown
        st.write("### Probability Breakdown")
        col_prob1, col_prob2 = st.columns(2)

        with col_prob1:
            st.metric("Probability to Stay", f"{stay_probability:.2f}%")

        with col_prob2:
            st.metric("Probability to Churn", f"{churn_probability:.2f}%")

        # Display a simple progress bar for visualization
        st.write("Churn Risk Level:")
        st.progress(churn_probability / 100)

    except Exception as e:
        st.error(f"Prediction failed: {e}")