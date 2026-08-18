# Machine Learning Assignment 2 — Classification Dashboard

## a. Problem Statement
Build and evaluate multiple classification models on one public classification dataset and deploy an interactive Streamlit application that allows a user to upload test data, select a model, and view evaluation metrics and classification results.

## b. Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic) dataset (UCI repository)

- Instances: 569
- Features: 30
- Task: Binary classification
- Target: `0 = Malignant`, `1 = Benign`
- Train/Test split: 80/20
- Random state: 42
- Test rows supplied in `test_data.csv`

The dataset satisfies the assignment minimum of 12 features and 500 instances.

## c. GitHub Repository Link
**Replace this placeholder with your actual GitHub repository URL after uploading the project:**

`PASTE_YOUR_GITHUB_REPOSITORY_LINK_HERE`

## d. Models Used

The assignment PDF lists five named models but later refers to six models. To remove this inconsistency, this implementation includes the five explicitly listed models plus **SVM as a sixth classifier**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| SVM | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy=0.9825, AUC=0.9954, F1=0.9861. The model provides a strong baseline for this binary classification problem. |
| Decision Tree | Accuracy=0.9211, AUC=0.9163, F1=0.9362. The model provides a strong baseline for this binary classification problem. |
| kNN | Accuracy=0.9737, AUC=0.9884, F1=0.9796. The model provides a strong baseline for this binary classification problem. |
| Naive Bayes | Accuracy=0.9386, AUC=0.9878, F1=0.9517. The model provides a strong baseline for this binary classification problem. |
| Random Forest | Accuracy=0.9474, AUC=0.9937, F1=0.9583. The model provides a strong baseline for this binary classification problem. |
| SVM | Accuracy=0.9825, AUC=0.9950, F1=0.9861. The model provides a strong baseline for this binary classification problem. |
| **Overall Winner** | **Logistic Regression** achieved the highest F1 score on the held-out test set used in this implementation. |

## Streamlit Application
The application provides:
1. CSV test-data upload
2. Model-selection dropdown
3. Accuracy, AUC, Precision, Recall, F1 and MCC
4. Confusion matrix
5. Classification report
6. Prediction table and downloadable predictions

## Project Structure
```text
ML_Assignment_2_Solution/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── svm.joblib
    ├── evaluation_results.csv
    └── metadata.json
```

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Link
**Replace this placeholder after deployment:**

`PASTE_YOUR_STREAMLIT_APP_LINK_HERE`

## Academic Integrity Note
The assignment instructions state that AI tools are allowed only for learning support and not for direct copy-paste submissions. Therefore, review, understand, customize, and test this implementation yourself before submission. Maintain your own Git commit history and record your BITS Virtual Lab execution screenshot as required.
