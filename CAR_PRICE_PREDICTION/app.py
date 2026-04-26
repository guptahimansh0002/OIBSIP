import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Set page config
st.set_page_config(page_title="Car Price Prediction App", layout="wide", page_icon="🚗")

# Define expected features to ensure consistency with the trained model
FEATURES = ['Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner', 'Car_Age', 'Kms_Per_Year']

# Robust artifact loading with error handling
@st.cache_resource
def load_model_artifacts():
    model_path = 'car_price_model.pkl'
    encoder_path = 'label_encoders.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None, f"Error: Model files not found. Please run the training notebook or script first."
    
    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        return model, encoders, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}. This often happens due to library version mismatches (e.g., scikit-learn version difference)."

# Initialize artifacts
model, encoders, error_msg = load_model_artifacts()

# App UI
st.title("🚗 Car Price Prediction App")
st.markdown("""
Predict the estimated selling price of a used car based on market features.
*Built by Himanshu Gupta | SRMU | Oasis Infobyte*
""")

if error_msg:
    st.error(error_msg)
    st.info("💡 Hint: Ensure you have run 'car_price_prediction.ipynb' and that all .pkl files are in the root directory.")
    st.stop()

# Sidebar inputs
st.sidebar.header("🔧 Input Car Details")

present_price = st.sidebar.slider("Present Showroom Price (in Lakhs)", 0.5, 50.0, 5.5, step=0.1)
year = st.sidebar.slider("Manufacturing Year", 2005, 2024, 2015)
driven_kms = st.sidebar.number_input("Total Kilometers Driven", min_value=0, max_value=500000, value=30000)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
selling_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# Pre-processing calculations
car_age = 2024 - year
kms_per_year = driven_kms / (car_age + 1)

# Prepare input data matching model feature order
input_df = pd.DataFrame([[
    present_price, 
    driven_kms, 
    fuel_type, 
    selling_type, 
    transmission, 
    owner, 
    car_age, 
    kms_per_year
]], columns=FEATURES)

# Encode categorical inputs using saved encoders
try:
    for col in ['Fuel_Type', 'Selling_type', 'Transmission']:
        input_df[col] = encoders[col].transform(input_df[col])
except Exception as e:
    st.error(f"Error in data transformation: {e}")
    st.stop()

# Prediction section
st.subheader("📊 Prediction Results")

if st.button("Calculate Predicted Price"):
    try:
        prediction = model.predict(input_df)[0]
        # Ensure prediction isn't negative
        predicted_price = max(0, prediction)
        depreciation = present_price - predicted_price
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Price", f"₹ {predicted_price:.2f} L")
        m2.metric("Total Depreciation", f"₹ {depreciation:.2f} L")
        m3.metric("Vehicle Age", f"{car_age} Yrs")
        
        # Visualization
        plot_df = pd.DataFrame({
            "Category": ["Present Price", "Predicted Selling Price"],
            "Value (Lakhs)": [present_price, predicted_price]
        })
        fig = px.bar(plot_df, x="Category", y="Value (Lakhs)", color="Category",
                     text_auto='.2f', title="Value Comparison",
                     color_discrete_map={"Present Price": "#3498db", "Predicted Selling Price": "#2ecc71"})
        st.plotly_chart(fig, use_container_width=True)
        
        if predicted_price == 0:
            st.warning("⚠️ The model predicts a negligible resale value based on the age and mileage provided.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("Adjust the sliders in the sidebar and click 'Calculate Predicted Price' to see the results.")

st.divider()
st.caption("Disclaimer: This prediction is based on historical data and should be used as an estimate only.")
