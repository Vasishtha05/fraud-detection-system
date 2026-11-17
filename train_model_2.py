
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

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

MODEL_DIR = PROJECT_ROOT / "model"

MODEL_PATH = (
    MODEL_DIR / "hist_gradient_boosting_model.pkl"
)

CONFUSION_MATRIX_PATH = (
    MODEL_DIR / "hist_gradient_boosting_confusion_matrix.txt"
)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment("Fraud Detection System")


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


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
print("Processed dataset shape:", df.shape)


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

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ==========================================
# HISTGRADIENTBOOSTING MODEL
# ==========================================

model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    class_weight="balanced",
    random_state=42
)


# ==========================================
# TRAIN AND TRACK WITH MLFLOW
# ==========================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with mlflow.start_run(
    run_name="HistGradientBoosting - Model 2"
):

    print("\nTraining HistGradientBoosting...")

    model.fit(X_train, y_train)

    print("Model training completed!")


    # ======================================
    # SAVE MODEL
    # ======================================

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("Model saved:", MODEL_PATH)


    # ======================================
    # PREDICTIONS
    # ======================================

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]


    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # ======================================
    # LOG PARAMETERS
    # ======================================

    mlflow.log_param(
        "model",
        "HistGradientBoostingClassifier"
    )

    mlflow.log_param(
        "learning_rate",
        0.1
    )

    mlflow.log_param(
        "max_iter",
        100
    )

    mlflow.log_param(
        "max_leaf_nodes",
        31
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )


    # ======================================
    # LOG METRICS
    # ======================================

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)


    # ======================================
    # LOG CONFUSION MATRIX
    # ======================================

    with open(
        CONFUSION_MATRIX_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Confusion Matrix - HistGradientBoosting\n"
        )

        file.write(str(cm))

    mlflow.log_artifact(
        str(CONFUSION_MATRIX_PATH)
    )


    # ======================================
    # LOG TRAINED MODEL
    # ======================================

    # Log the trained model as a file artifact
    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="trained_model"
    )

    print("Trained model successfully saved as MLflow artifact!")
    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\nHISTGRADIENTBOOSTING RESULTS")
    print("--------------------------------")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)
    print("PR-AUC   :", pr_auc)

    print("\nConfusion Matrix:")
    print(cm)


print("\nModel 2 successfully logged to MLflow!")
print("Experiment: Fraud Detection System")
print("Tracking database:", MLFLOW_DB)
