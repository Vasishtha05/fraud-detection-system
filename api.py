
import os
import logging

import mlflow

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from src.predict import predict_transaction
from src.database import create_table, get_recent_predictions


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# MLFLOW CONFIGURATION
# ==========================================

# Docker API connects to the MLflow server running
# on the Windows host through host.docker.internal.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000"
)

MLFLOW_EXPERIMENT_NAME = "Fraud Detection System"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Fraud detection API with MLflow prediction "
        "tracing and SQLite prediction history."
    ),
    version="2.0.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:63342",
        "http://127.0.0.1:63342",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ==========================================
# REQUEST SCHEMA
# ==========================================

class TransactionRequest(BaseModel):

    step: int = Field(ge=0)

    type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    amount: float = Field(ge=0)

    isFlaggedFraud: int = Field(ge=0, le=1)


# ==========================================
# MLFLOW TRACED PREDICTION
# ==========================================

@mlflow.trace(name="fraud_transaction_prediction")
def traced_prediction(transaction: dict):

    # MLflow automatically records:
    # 1. Function inputs
    # 2. Function outputs
    # 3. Execution duration
    # 4. Exceptions, if prediction fails

    result = predict_transaction(transaction)

    return result


# ==========================================
# INITIALIZE DATABASE AND MLFLOW
# ==========================================

@app.on_event("startup")
def startup_event():

    create_table()

    try:
        mlflow.set_experiment(
            MLFLOW_EXPERIMENT_NAME
        )

        logger.info(
            "MLflow connected: %s",
            MLFLOW_TRACKING_URI
        )

        logger.info(
            "MLflow experiment: %s",
            MLFLOW_EXPERIMENT_NAME
        )

    except Exception:
        logger.exception(
            "MLflow unavailable. API will continue without tracing."
        )


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
        "mlflow": MLFLOW_TRACKING_URI,
        "experiment": MLFLOW_EXPERIMENT_NAME
    }


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Fraud Detection API"
    }


# ==========================================
# FRAUD PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(request: TransactionRequest):

    try:

        transaction = request.model_dump()

        # Ensure the correct MLflow experiment is active.
        try:
            mlflow.set_experiment(
                MLFLOW_EXPERIMENT_NAME
            )
        except Exception:
            logger.exception(
                "Could not select MLflow experiment."
            )

        # Run and trace the prediction.
        result = traced_prediction(transaction)

        logger.info(
            "Prediction completed successfully."
        )

        return {
            "status": "success",
            "transaction": transaction,
            "prediction": result
        }

    except (ValueError, FileNotFoundError) as exc:

        logger.warning(
            "Prediction validation error: %s",
            exc
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception as exc:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs."
        ) from exc


# ==========================================
# RECENT PREDICTIONS ENDPOINT
# ==========================================

@app.get("/predictions")
def recent_predictions(limit: int = 10):

    if not 1 <= limit <= 100:

        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100"
        )

    records = get_recent_predictions(limit)

    return {
        "count": len(records),
        "predictions": records
    }
