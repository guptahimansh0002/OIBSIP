# Iris Flower Classification

## Project Overview

This project is a complete **machine learning classification notebook** built using Python and Jupyter Notebook as part of the **Oasis Infobyte Data Science Internship**.

The goal of the project is to classify Iris flowers into their species based on flower measurements using multiple machine learning algorithms and a structured data science workflow.

## Internship Details

- **Role:** Data Science Intern
- **Organization:** Oasis Infobyte
- **Project Title:** Iris Flower Classification

## Dataset Used

- **Folder:** `iris_data`
- **File:** `Iris.csv`
- **Total Records:** 150
- **Target Column:** `Species`

The dataset contains flower measurement features such as:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

It also contains an `Id` column, which was identified as a non-predictive identifier and excluded during model building.

## What I Did in `iris.ipynb`

The notebook follows a complete end-to-end machine learning pipeline:

### 1. Imported Required Libraries

I imported the libraries needed for:

- Data handling with `pandas` and `numpy`
- Data visualization with `matplotlib` and `seaborn`
- Machine learning with `scikit-learn`

### 2. Loaded the Dataset

I loaded the dataset directly from the local `iris_data/Iris.csv` file and verified the structure of the data.

### 3. Performed Exploratory Data Analysis (EDA)

I explored the dataset by checking:

- Shape of the dataset
- Data types
- Summary statistics
- Missing values
- Duplicate values
- Class distribution

### 4. Visualized the Data

I created multiple plots to better understand the dataset:

- Species count plot
- Feature histograms
- Boxplots by species
- Pairplot for feature relationships
- Correlation heatmap

### 5. Preprocessed the Data

I prepared the data for modeling by:

- Removing the `Id` column
- Splitting features and target
- Applying an 80:20 train-test split
- Using scaling where needed
- Building preprocessing pipelines for reproducibility

### 6. Built Multiple Machine Learning Models

I trained and tested the following models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

### 7. Evaluated Model Performance

I evaluated the models using:

- Accuracy score
- Classification report
- Confusion matrix

### 8. Compared the Models

I created a comparison table and chart to compare model accuracy and identify the best-performing approach for this dataset.

### 9. Added Final Conclusion

I summarized the key findings from the project, explained model performance, and highlighted why the project is a good example of practical machine learning and data analysis.

### 10. Saved the Best Model for Deployment

I added a model-saving step in the notebook using `joblib`.

- The selected best model is retrained on the full dataset
- The trained pipeline is saved as `best_iris_model.joblib`
- This saved model is later used in the Streamlit application for prediction

## Key Findings

- The dataset is clean and balanced.
- There are no missing values or duplicate rows.
- Petal features are more useful than sepal features for separating flower species.
- `Iris-setosa` is clearly distinguishable from the other two classes.
- Most classification confusion happens between `Iris-versicolor` and `Iris-virginica`.
- All three models achieved strong performance on the test data.

## Best Model Result

In the current notebook run:

- **Best Model:** Logistic Regression
- **Accuracy:** 93.33%

Note: All three models achieved the same accuracy in this notebook execution, but Logistic Regression appears first in the sorted comparison.

## Project Files

- `iris.ipynb` - Main Jupyter Notebook with full workflow
- `app.py` - Streamlit web application for Iris species prediction
- `best_iris_model.joblib` - Saved best machine learning model from the notebook
- `iris_data/Iris.csv` - Dataset used in the project
- `requirements.txt` - Required Python libraries
- `README.md` - Project documentation

## Streamlit Application

I also created a **Streamlit app** to make the project interactive and deployment-ready.

### What the app does

- Loads the saved model file `best_iris_model.joblib`
- Accepts flower measurements from the user
- Predicts the Iris species
- Displays prediction confidence when available

This helps demonstrate how a machine learning notebook can be extended into a simple real-world application.

## How to Run This Project

### 1. Open the project folder

```bash
cd IRIS_FLOWER_CLASSIFICATION
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook iris.ipynb
```

Then run all notebook cells from top to bottom.

This will:

- load the dataset
- train and compare models
- save the best model as `best_iris_model.joblib`

### 4. Run the Streamlit app

After the model file is available, run:

```bash
streamlit run app.py
```

## How I Run It

My normal workflow for this project is:

1. Open `iris.ipynb`
2. Run all cells in order
3. Make sure `best_iris_model.joblib` is created in the project folder
4. Start the Streamlit app using `streamlit run app.py`
5. Enter flower measurements in the app and check the predicted species

## Example App Input

You can test the app using a common Iris-setosa example:

- Sepal Length: 5.1
- Sepal Width: 3.5
- Petal Length: 1.4
- Petal Width: 0.2

Expected prediction:

- `Iris-setosa`

## Skills Demonstrated

This project demonstrates my skills in:

- Data cleaning and exploration
- Data visualization
- Feature understanding
- Machine learning model building
- Model evaluation
- Comparison-based problem solving
- Writing clean, beginner-friendly, and structured analysis

## Tools and Technologies

- Python
- Jupyter Notebook
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Conclusion

This project shows a full beginner-to-intermediate level machine learning workflow in a clean and professional format. It highlights my ability to analyze data, build and compare models, interpret outputs, and communicate results clearly through both code and explanation.
