
import os
import sys
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------------
# PATHS
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "PS_20174392719_1491204439457_log.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hist_gradient_boosting_model.pkl"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# -----------------------------------
# FEATURE ENGINEERING
# Match train_model_2.py
# -----------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from feature_engineering import create_features

df = df.drop(columns=["nameOrig", "nameDest"])
df = create_features(df)

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Exclude balance features, matching Model 2
balance_columns = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X = X.drop(
    columns=balance_columns,
    errors="ignore"
)

# -----------------------------------
# SAME TEST SPLIT
# -----------------------------------
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Test data prepared:", X_test.shape)

# -----------------------------------
# LOAD SAVED MODEL
# -----------------------------------
print("Loading HistGradientBoosting model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# -----------------------------------
# FRAUD PROBABILITIES
# -----------------------------------
print("Generating fraud probabilities...")

y_probability = model.predict_proba(X_test)[:, 1]

# -----------------------------------
# MLflow EXPERIMENT
# -----------------------------------
mlflow.set_experiment("Fraud Detection System")

thresholds = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90
]

results = []

print("\nMODEL 2 THRESHOLD RESULTS")
print("-" * 90)

for threshold in thresholds:

    y_pred = (y_probability >= threshold).astype(int)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    alert_rate = y_pred.mean()

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "alert_rate": alert_rate
    })

    with mlflow.start_run(
        run_name=f"HistGradientBoosting_threshold_{threshold:.2f}"
    ):
        mlflow.log_param("model", "HistGradientBoosting")
        mlflow.log_param("threshold", threshold)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("true_positives", int(tp))
        mlflow.log_metric("true_negatives", int(tn))
        mlflow.log_metric("alert_rate", alert_rate)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"FP: {fp:,} | "
        f"FN: {fn:,}"
    )

# -----------------------------------
# SAVE RESULTS
# -----------------------------------
results_df = pd.DataFrame(results)

results_path = os.path.join(
    BASE_DIR,
    "model",
    "model2_threshold_results.csv"
)

results_df.to_csv(results_path, index=False)

print("\nResults saved to:", results_path)

print("\nBEST F1 THRESHOLD")
print(
    results_df.loc[
        results_df["f1_score"].idxmax()
    ].to_string()
)

print("\nThreshold tuning completed successfully!")
