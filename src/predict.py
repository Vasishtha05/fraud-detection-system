
import os
import mlflow
import sys
import joblib
import pandas as pd
import numpy as np
from src.database import create_table, save_prediction

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if mlflow_uri:
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("Fraud Detection System")
    except Exception as e:
        print(f"MLflow setup skipped: {e}")
# -----------------------------------
# PROJECT PATHS
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hist_gradient_boosting_model.pkl"
)

# Experimental threshold from our tuning experiments
FRAUD_THRESHOLD = 0.90

# Make src modules importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import create_features


# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    print("HistGradientBoosting model loaded successfully!")

    return model


# -----------------------------------
# PREPARE TRANSACTION FEATURES
# -----------------------------------
def prepare_transaction(transaction, model):
    """
    transaction must contain:
    step, type, amount, isFlaggedFraud

    Balance columns and transaction identifiers are not required.
    """

    required_columns = [
        "step",
        "type",
        "amount",
        "isFlaggedFraud"
    ]

    missing = [
        column for column in required_columns
        if column not in transaction
    ]

    if missing:
        raise ValueError(
            f"Missing required transaction fields: {missing}"
        )

    if transaction["type"] not in [
        "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
    ]:
        raise ValueError("Invalid transaction type.")

    if float(transaction["step"]) < 0:
        raise ValueError("step must be non-negative.")

    if float(transaction["amount"]) < 0:
        raise ValueError("amount must be non-negative.")

    if transaction["isFlaggedFraud"] not in [0, 1]:
        raise ValueError("isFlaggedFraud must be 0 or 1.")

    # Match training feature construction
    df = pd.DataFrame([{
        "step": int(transaction["step"]),
        "type": transaction["type"],
        "amount": float(transaction["amount"]),
        "isFlaggedFraud": int(transaction["isFlaggedFraud"])
    }])

    df = create_features(df)

    # Ensure exactly the same feature order as model training
    expected_features = list(model.feature_names_in_)

    missing_features = [
        col for col in expected_features if col not in df.columns
    ]

    for col in missing_features:
        df[col] = 0

    extra_features = [
        col for col in df.columns if col not in expected_features
    ]

    df = df.drop(columns=extra_features, errors="ignore")
    df = df[expected_features]

    return df


_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


# -----------------------------------
# PREDICT FRAUD
# -----------------------------------
@mlflow.trace(name="FraudGuard Transaction Prediction")
def predict_transaction(transaction):
    model = get_model()
    X = prepare_transaction(transaction, model)

    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= FRAUD_THRESHOLD)

    result = {
        "prediction": prediction,
        "result": "FRAUD" if prediction else "LEGITIMATE",
        "fraud_probability": round(probability, 6),
        "fraud_probability_percent": round(probability * 100, 2),
        "threshold": FRAUD_THRESHOLD
    }

    # Initialize the table and save this prediction
    create_table()
    record_id = save_prediction(transaction, result)
    result["record_id"] = record_id

    return result



# -----------------------------------
# SAMPLE RUN
# -----------------------------------
if __name__ == "__main__":

    # Synthetic example for testing the prediction pipeline.
    sample_transaction = {
        "step": 10,
        "type": "TRANSFER",
        "amount": 5000.0,
        "isFlaggedFraud": 0
    }

    result = predict_transaction(sample_transaction)

    print("\nTransaction prediction")
    print("-" * 35)

    for key, value in result.items():
        print(f"{key}: {value}")
