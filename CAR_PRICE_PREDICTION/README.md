# 🚗 Car Price Prediction Project
**Author:** Himanshu Gupta | SRMU | Oasis Infobyte Internship

This repository contains a complete end-to-end Machine Learning project to predict the selling price of used cars. It includes data cleaning, exploratory analysis, model building using Random Forest, and a deployment-ready Streamlit web application.

---

## 📋 Project Overview
The goal of this project is to estimate the resale value of a car based on several parameters like its original price, kilometers driven, age, fuel type, and transmission. This helps both buyers and sellers in determining a fair market price.

## 📊 Dataset Description
The data is sourced from `car_data/car data.csv`. Key features include:
- **Car_Name**: Name/Model of the car.
- **Year**: Manufacturing year.
- **Selling_Price**: Target variable (Price at which the car is being sold).
- **Present_Price**: Current showroom price of the car.
- **Driven_kms**: Total distance covered in kilometers.
- **Fuel_Type**: Petrol, Diesel, or CNG.
- **Selling_type**: Dealer or Individual.
- **Transmission**: Manual or Automatic.
- **Owner**: Number of previous owners.

## 🔍 Key Insights from Analysis
- **Depreciation**: Car Age has a strong negative correlation with Selling Price.
- **Price Drivers**: The 'Present Price' is the most significant predictor of the resale value.
- **Fuel Impact**: Diesel cars generally command a higher resale value than Petrol cars.
- **Transmission**: Automatic cars tend to be priced higher than manual ones.

## 🛠️ Technical Stack
- **Data Handling**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-Learn (Random Forest Regressor)
- **Deployment**: Streamlit
- **Environment**: Jupyter Notebook, Python 3.9+

## 🚀 How to Run the Project

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Explore the Analysis
Open the Jupyter Notebook to view the full data exploration and model training process:
```bash
jupyter notebook car_price_prediction.ipynb
```

### 3. Launch the Web App
Run the Streamlit application to predict car prices interactively:
```bash
streamlit run app.py
```

## 🏗️ Development Journey: How this project was built
According to the implementation logic followed, here is a summary of the steps taken to deliver this project:

1.  **Automated Data Inspection**: The project began with an automated scan of the raw `car data.csv` to identify the actual column names and data types, ensuring the entire pipeline uses real-time data attributes rather than assumptions.
2.  **Structured Notebook Generation**: Instead of a simple script, I used the `nbformat` library to programmatically construct a professional Jupyter Notebook. This ensures a consistent structure where every code block is preceded by a markdown description and followed by a data-driven observation.
3.  **Surgical Feature Engineering**: I implemented specific transformations to make the model smarter:
    *   **Car_Age**: Derived from the manufacturing year to capture depreciation.
    *   **Kms_Per_Year**: Calculated to understand the intensity of car usage.
    *   **Brand Extraction**: Extracted the manufacturer name from the model string to capture brand value.
4.  **Robust Model Pipeline**: I chose a **Random Forest Regressor** for its ability to handle non-linear relationships and its resistance to overfitting. I also implemented a `LabelEncoder` persistence strategy, saving the encoders as `.pkl` files so the web app can transform user inputs exactly like the training data.
5.  **Interactive Deployment**: I built a high-performance Streamlit dashboard. It doesn't just show a number; it provides context by calculating depreciation and visualizing the price gap between the original and predicted values using Plotly.
6.  **Professional Formatting**: Finally, I refined the notebook's aesthetics using horizontal rules, emojis, and blockquotes to make the technical findings accessible and easy to read.

## 📂 File Structure
- `car_data/`: Contains the raw dataset.
- `car_price_prediction.ipynb`: Comprehensive EDA and model training notebook.
- `app.py`: Streamlit application code.
- `car_price_model.pkl`: Pre-trained Random Forest model.
- `label_encoders.pkl`: Saved encoders for categorical features.
- `requirements.txt`: List of required Python libraries.
- `README.md`: Project documentation.

---
*Developed as part of the Oasis Infobyte Data Science Internship.*
