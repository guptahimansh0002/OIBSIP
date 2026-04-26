<<<<<<< HEAD
# Unemployment Analysis in India (Oasis Infobyte Project)

## Project Overview
This project performs a comprehensive Data Science analysis on the unemployment trends in India, specifically focusing on the period surrounding the COVID-19 pandemic. It explores how various factors like geography (States/Regions), sector (Urban vs. Rural), and time (Date) influenced the unemployment rate.

**Author:** Himanshu Gupta  
**Task:** Data Science Internship at Oasis Infobyte

---

## Project Structure
- `unemployment_analysis.ipynb`: The core analysis notebook containing data cleaning, EDA, and ML training.
- `app.py`: A Streamlit-based web interface for real-time unemployment rate prediction.
- `unemployment_data/`: Directory containing the source dataset (`Unemployment in India.csv`).
- `requirements.txt`: List of Python libraries required to run the project.
- `unemployment_model.pkl`: The trained Random Forest Regressor model.
- `region_encoder.pkl`: Label encoder for categorical region data.

---

## Step-by-Step Implementation

### Step 1: Data Acquisition & Research
We started by loading the `Unemployment in India.csv` dataset and inspecting its structure. This involved identifying column names, data types, and checking for missing values.

### Step 2: Data Cleaning & Preprocessing
- **Whitespace Removal:** Column names had leading/trailing spaces which were stripped for easier access.
- **Handling Nulls:** 28 empty rows at the end of the file were removed, and remaining missing values were filled with the median.
- **Date Parsing:** The 'Date' column was converted to datetime objects.
- **Feature Engineering:** Extracted `Month`, `Year`, and `Quarter` from the date to capture seasonal and temporal trends.

### Step 3: Exploratory Data Analysis (EDA)
- **Time Series Analysis:** Visualized national unemployment trends to identify the massive spike during the 2020 lockdowns.
- **State-wise Comparison:** Analyzed which states (like Haryana and Tripura) faced the highest unemployment.
- **Urban vs. Rural Analysis:** Compared the economic resilience of different sectors.
- **Phase Analysis:** Categorized data into Pre-Covid, During-Covid, and Post-Covid to quantify the pandemic's impact.

### Step 4: Machine Learning Modeling
- **Encoding:** Categorical variables (Region and Area) were converted to numerical formats using Label Encoding.
- **Model Selection:** A **Random Forest Regressor** was chosen for its ability to handle non-linear relationships and provide feature importance.
- **Training:** The data was split (80/20) and the model was trained to predict the `Estimated Unemployment Rate (%)`.
- **Validation:** Performance was evaluated using MAE, RMSE, and R2 Score.

### Step 5: Web UI Development (Streamlit)
Finally, we developed a user interface using Streamlit that allows users to select a State, an Area (Urban/Rural), and a Date to get an instant prediction of the unemployment rate based on the trained model.

---

## How to Run the Project

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis (Optional)
If you want to re-generate the models or explore the data:
```bash
jupyter nbconvert --to notebook --execute unemployment_analysis.ipynb
```

### 3. Launch the Streamlit App
To start the prediction interface:
```bash
streamlit run app.py
```

---

## Key Findings
- **Peak Impact:** The highest unemployment spike occurred in April-May 2020.
- **Sector Vulnerability:** Urban areas consistently showed higher volatility during the pandemic compared to rural areas.
- **Predictive Power:** The Region (State) is the most significant predictor of unemployment rates in this dataset.
=======
# OIBSIP - Data Science Internship Projects

This repository contains projects completed during my Data Science Internship at Oasis Infobyte (OIBSIP).

## 📌 Projects

1. Iris Flower Classification  
2. Unemployment Analysis  
3. Car Price Prediction  
4. Email Spam Detection  
5. Sales Prediction 
>>>>>>> 50276c2902c9ecad949629952794d2287a02fabe
