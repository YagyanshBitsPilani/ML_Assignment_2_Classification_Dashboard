import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer ML Classifier", page_icon="🧠", layout="wide")

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
}

FEATURES = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

st.title("🧠 Breast Cancer Classification Dashboard")
st.caption("M.Tech (AIML/DSE) — Machine Learning Assignment 2")

st.info(
    "Upload the test CSV supplied with this project. The file must contain the 30 feature columns "
    "and a target column named 'target'."
)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

col1, col2 = st.columns(2)
with col1:
    selected_model = st.selectbox("Select classification model", list(MODEL_FILES.keys()))
with col2:
    st.write("Dataset: UCI Breast Cancer Wisconsin (Diagnostic)")
    st.write("30 features • 569 instances • binary classification")

@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])

def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1 Score": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, metrics

if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error("Missing feature columns: " + ", ".join(missing))
        st.stop()
    if "target" not in df.columns:
        st.error("The uploaded test file must contain a 'target' column.")
        st.stop()

    X = df[FEATURES]
    y = df["target"].astype(int)

    model = load_model(selected_model)
    pred, metrics = evaluate(model, X, y)

    st.subheader(f"Results — {selected_model}")
    metric_cols = st.columns(6)
    for c, (label, value) in zip(metric_cols, metrics.items()):
        c.metric(label, f"{value:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Malignant (0)", "Actual Benign (1)"],
        columns=["Predicted Malignant (0)", "Predicted Benign (1)"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(
        y, pred, target_names=["Malignant", "Benign"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    output = df.copy()
    output["predicted_target"] = pred
    output["predicted_label"] = np.where(pred == 1, "Benign", "Malignant")
    st.subheader("Prediction Output")
    st.dataframe(output.head(20), use_container_width=True)

    st.download_button(
        "Download predictions CSV",
        output.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv"
    )
else:
    st.warning("Please upload test_data.csv to view model evaluation results.")
