import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Unemployment Rate Predictor", page_icon="📈", layout="centered")

# Load model and encoder
@st.cache_resource
def load_assets():
    model = joblib.load('unemployment_model.pkl')
    le_region = joblib.load('region_encoder.pkl')
    return model, le_region

try:
    model, le_region = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Please run the analysis notebook first to generate 'unemployment_model.pkl' and 'region_encoder.pkl'.")
    st.stop()

# Header
st.title("📈 Unemployment Rate Predictor")
st.markdown("""
This app predicts the **Estimated Unemployment Rate (%)** in India based on Region, Area, and Date.
Data is processed through a Random Forest Regressor trained on historical trends (2019-2020).
""")

# Sidebar for inputs
st.sidebar.header("Input Features")

regions = ['Andhra Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Meghalaya', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']

selected_region = st.sidebar.selectbox("Region/State", regions)
selected_area = st.sidebar.selectbox("Area", ["Urban", "Rural"])
selected_date = st.sidebar.date_input("Select Date", datetime(2020, 5, 1))

# Feature engineering for prediction
month = selected_date.month
year = selected_date.year
quarter = (month - 1) // 3 + 1

# Encode inputs
region_encoded = le_region.transform([selected_region])[0]
area_encoded = 1 if selected_area == "Urban" else 0 # Based on typical LabelEncoder alphabetics (Rural=0, Urban=1)

# Prediction button
if st.button("Predict Unemployment Rate"):
    features = np.array([[month, year, quarter, region_encoded, area_encoded]])
    prediction = model.predict(features)[0]
    
    # UI Output
    st.success(f"### Predicted Unemployment Rate: {prediction:.2f}%")
    
    # Contextual feedback
    if prediction > 15:
        st.warning("Critical: This is a high unemployment rate prediction.")
    elif prediction > 8:
        st.info("Moderate: Unemployment rate is above national average.")
    else:
        st.info("Stable: Unemployment rate is within manageable limits.")

# Footer
st.markdown("---")
st.markdown("Developed by **Himanshu Gupta** | Oasis Infobyte Data Science Project")
