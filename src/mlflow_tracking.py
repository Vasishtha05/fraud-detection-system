
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from feature_engineering import create_features


# ==========================================
# PROJECT PATHS AND MLFLOW CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

MODEL_PATH = PROJECT_ROOT / "model" / "fraud_model.pkl"
SCALER_PATH = PROJECT_ROOT / "model" / "scaler.pkl"

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT / "model" / "confusion_matrix.txt"
)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment("Fraud Detection System")


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading fraud detection dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# REMOVE IDENTIFIERS
# ==========================================

df = df.drop(
    columns=["nameOrig", "nameDest"],
    errors="ignore"
)


# ==========================================
# FEATURE ENGINEERING
# ==========================================

df = create_features(df)

print("Feature engineering completed!")


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# ==========================================
# REMOVE BALANCE FEATURES
# ==========================================

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


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# LOAD EXISTING MODEL AND SCALER
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model and scaler loaded successfully!")


# ==========================================
# SCALE TEST DATA
# ==========================================

X_test_scaled = scaler.transform(X_test)


# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# ==========================================
# CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

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

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

cm = confusion_matrix(y_test, y_pred)


# ==========================================
# MLFLOW EXPERIMENT RUN
# ==========================================

with mlflow.start_run(
    run_name="Logistic Regression - Baseline"
):

    # Parameters
    mlflow.log_param(
        "model",
        "Logistic Regression"
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "feature_engineering",
        "hour, day, amount_log"
    )

    mlflow.log_param(
        "balance_features_removed",
        True
    )

    # Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    # Confusion matrix artifact
    CONFUSION_MATRIX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CONFUSION_MATRIX_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Confusion Matrix - Logistic Regression\n"
        )

        file.write(str(cm))

    mlflow.log_artifact(
        str(CONFUSION_MATRIX_PATH)
    )

    # Log existing model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="fraud_logistic_regression"
    )

    print("\nMLflow run completed successfully!")

    print("\nLOGISTIC REGRESSION RESULTS")
    print("--------------------------------")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)
    print("PR-AUC   :", pr_auc)

    print("\nConfusion Matrix:")
    print(cm)


print("\nLogistic Regression successfully logged to MLflow!")
print("Experiment: Fraud Detection System")
print("Tracking database:", MLFLOW_DB)
