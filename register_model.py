
import os
import joblib
import mlflow
import mlflow.sklearn

# -----------------------------------
# PROJECT PATHS
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hist_gradient_boosting_model.pkl"
)

# -----------------------------------
# CONFIGURATION
# -----------------------------------
MODEL_NAME = "FraudDetectionHistGradientBoosting"
SELECTED_THRESHOLD = 0.90

# Use the same MLflow tracking location as your existing scripts.


# Use MLflow's default tracking location
mlflow.set_experiment("Fraud Detection System")

mlflow.set_experiment("Fraud Detection System")

# -----------------------------------
# LOAD MODEL
# -----------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("HistGradientBoosting model loaded.")

# -----------------------------------
# REGISTER MODEL
# -----------------------------------
with mlflow.start_run(run_name="Final_Model_Registration"):

    mlflow.log_param("model_type", "HistGradientBoosting")
    mlflow.log_param("decision_threshold", SELECTED_THRESHOLD)
    mlflow.log_param("threshold_status", "experimental_candidate")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="final_fraud_detection_model",
        serialization_format="pickle",
        registered_model_name=MODEL_NAME
    )

    mlflow.set_tag("project", "Fraud Detection System")
    mlflow.set_tag("stage", "candidate")

    print("Model logged and registration requested.")
    print("Model URI:", model_info.model_uri)

print("Registration script completed.")
print("Check the MLflow UI for registration status.")
