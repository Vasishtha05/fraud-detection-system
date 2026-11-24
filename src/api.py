
import os
import logging
from pathlib import Path

import mlflow

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from src.predict import predict_transaction
from src.database import create_table, get_recent_predictions


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# MLFLOW CONFIGURATION
# ==========================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "Fraud Detection System"

if MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except Exception:
        logger.exception("Failed to set MLflow tracking URI.")


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    if MLFLOW_TRACKING_URI:
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
    else:
        logger.info("MLFLOW_TRACKING_URI not set. Running in standalone mode.")


# ==========================================
# FRONTEND & ROOT ENDPOINTS
# ==========================================

@app.get("/")
def home():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health",
        "mlflow": MLFLOW_TRACKING_URI or "disabled",
        "experiment": MLFLOW_EXPERIMENT_NAME
    }


@app.get("/style.css")
def serve_css():
    css_path = FRONTEND_DIR / "style.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/script.js")
def serve_js():
    js_path = FRONTEND_DIR / "script.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found")


@app.get("/api")
def api_info():
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health",
        "mlflow": MLFLOW_TRACKING_URI or "disabled",
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

        if MLFLOW_TRACKING_URI:
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
