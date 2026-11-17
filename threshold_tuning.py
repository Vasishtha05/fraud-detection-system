import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from feature_engineering import create_features


# --------------------------------
# LOAD DATASET
# --------------------------------

file_path = "../data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")


# --------------------------------
# REMOVE IDENTIFIERS
# --------------------------------

df = df.drop(columns=["nameOrig", "nameDest"])


# --------------------------------
# FEATURE ENGINEERING
# --------------------------------

df = create_features(df)


# --------------------------------
# FEATURES AND TARGET
# --------------------------------

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# --------------------------------
# REMOVE BALANCE FEATURES
# --------------------------------

balance_columns = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X = X.drop(columns=balance_columns)


# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# LOAD MODEL AND SCALER
# --------------------------------

model = joblib.load("../model/fraud_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

print("Model and scaler loaded successfully!")


# --------------------------------
# SCALE TEST DATA
# --------------------------------

X_test_scaled = scaler.transform(X_test)


# --------------------------------
# FRAUD PROBABILITIES
# --------------------------------

fraud_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# --------------------------------
# MLflow EXPERIMENT
# --------------------------------

mlflow.set_experiment("Fraud Detection System")


# --------------------------------
# TEST DIFFERENT THRESHOLDS
# --------------------------------

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


for threshold in thresholds:

    print("\n-----------------------------")
    print("Threshold:", threshold)
    print("-----------------------------")

    # Convert probability into prediction
    y_pred = (
        fraud_probability >= threshold
    ).astype(int)

    # Metrics
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    total_transactions = len(y_test)

    alert_rate = (
        (fp + tp) / total_transactions
    )

    # --------------------------------
    # MLflow Run
    # --------------------------------

    with mlflow.start_run(
        run_name=f"Threshold {threshold}"
    ):

        mlflow.log_param(
            "model",
            "Logistic Regression"
        )

        mlflow.log_param(
            "threshold",
            threshold
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "false_positives",
            fp
        )

        mlflow.log_metric(
            "false_negatives",
            fn
        )

        mlflow.log_metric(
            "true_positives",
            tp
        )

        mlflow.log_metric(
            "true_negatives",
            tn
        )

        mlflow.log_metric(
            "alert_rate",
            alert_rate
        )

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("False Positives:", fp)
    print("False Negatives:", fn)

print("\nThreshold experiments completed successfully!")