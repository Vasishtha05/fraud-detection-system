import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from feature_engineering import create_features


# -----------------------------
# LOAD DATASET
# -----------------------------

file_path = "../data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")


# -----------------------------
# REMOVE IDENTIFIERS
# -----------------------------

df = df.drop(columns=["nameOrig", "nameDest"])


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

df = create_features(df)


# -----------------------------
# SEPARATE FEATURES AND TARGET
# -----------------------------

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# -----------------------------
# REMOVE BALANCE FEATURES
# -----------------------------

balance_columns = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X = X.drop(columns=balance_columns)


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# LOAD MODEL AND SCALER
# -----------------------------

model = joblib.load("../model/fraud_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

print("Model and scaler loaded successfully!")


# -----------------------------
# SCALE TEST DATA
# -----------------------------

X_test_scaled = scaler.transform(X_test)


# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]


# -----------------------------
# CONFUSION MATRIX
# -----------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# ROC-AUC
# -----------------------------

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC:")
print(roc_auc)


# -----------------------------
# PRECISION-RECALL AUC
# -----------------------------

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\nPR-AUC:")
print(pr_auc)


print("\nEvaluation completed successfully!")