# frontend/app.py

import streamlit as st
import requests
import json

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("Customer Churn Prediction")

FASTAPI_URL = st.sidebar.text_input(
    "FastAPI URL",
    value=st.secrets.get("FASTAPI_URL", "http://fastapi:80"),
    help="URL of the FastAPI backend"
)

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 50.0)
        total_charges = st.slider("Total Charges", 0.0, 9000.0, 1000.0)
    
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        st.success("Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", result["prediction"])
        
        with col2:
            churn_prob = result["churn_probability"] * 100
            st.metric("Churn Probability", f"{churn_prob:.2f}%")
        
        with col3:
            risk_color = "🔴" if churn_prob > 70 else "🟡" if churn_prob > 50 else "🟢"
            st.metric("Risk Level", risk_color)
        
        # Progress bar
        st.progress(result["churn_probability"])
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
