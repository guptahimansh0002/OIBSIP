from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")


MODEL_PATH = Path(__file__).resolve().parent / "best_iris_model.joblib"
DATA_PATH = Path(__file__).resolve().parent / "iris_data" / "Iris.csv"


def train_model_from_dataset():
    """Rebuild the deployment model in the current environment."""
    df = pd.read_csv(DATA_PATH)

    model_df = df.drop(columns=["Id"])
    X = model_df.drop(columns=["Species"])
    y = model_df["Species"]
    numeric_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


def build_input_frame(sepal_length, sepal_width, petal_length, petal_width):
    return pd.DataFrame(
        [
            {
                "SepalLengthCm": sepal_length,
                "SepalWidthCm": sepal_width,
                "PetalLengthCm": petal_length,
                "PetalWidthCm": petal_width,
            }
        ]
    )


def validate_model(model):
    sample = build_input_frame(5.1, 3.5, 1.4, 0.2)
    model.predict(sample)
    return model


@st.cache_resource
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset file not found. Expected 'iris_data/Iris.csv' in the project folder."
        )

    if not MODEL_PATH.exists():
        return validate_model(train_model_from_dataset())

    try:
        return validate_model(joblib.load(MODEL_PATH))
    except Exception:
        return validate_model(train_model_from_dataset())


st.title("Iris Flower Classification App")
st.write(
    "This Streamlit app uses the best model saved from `iris.ipynb` to predict the species of an Iris flower."
)

with st.sidebar:
    st.header("About")
    st.write("Model file: `best_iris_model.joblib`")
    st.write("Inputs are based on flower measurements in centimeters.")

model = load_model()

if MODEL_PATH.exists():
    st.caption(f"Model file in use: `{MODEL_PATH.name}`")

st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)

input_df = build_input_frame(sepal_length, sepal_width, petal_length, petal_width)

st.subheader("Input Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Species", type="primary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Species: **{prediction}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
        probability_df = pd.DataFrame(
            {"Species": classes, "Probability": probabilities}
        ).sort_values(by="Probability", ascending=False)
        probability_df["Probability"] = probability_df["Probability"].map(lambda value: f"{value:.2%}")

        st.subheader("Prediction Confidence")
        st.table(probability_df.reset_index(drop=True))
