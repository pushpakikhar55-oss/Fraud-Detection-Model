"""
FastAPI Application for Real-Time Financial Fraud Detection & Analytics.
Exposes REST endpoints for prediction, transaction ingestion, alert triage,
portfolio analytics, and serves the interactive monitoring dashboard.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd

from src.api.models import (
    TransactionSubmitRequest,
    PredictRequest,
    PredictResponse,
    AlertUpdateRequest,
    AnalyticsSummary,
)
from src.api.database import DatabaseManager
from src.api.service import InferenceService

logger = logging.getLogger("fraud_platform.api")

app = FastAPI(
    title="AI Financial Fraud Detection Platform",
    description="Enterprise Fraud Risk Scoring, Anomaly Detection & Explainable AI REST Service",
    version="1.0.0",
)

# Keep the default browser policy local-only; deployments can add their own
# comma-separated origins through FRAUD_CORS_ORIGINS.
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRAUD_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = DatabaseManager()
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)


@app.middleware("http")
async def add_request_observability(request, call_next):
    """Attach a traceable request ID and log method, path, status, and latency."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Fraud Detection ML & Risk Scoring API",
        "version": "1.0.0",
    }


@app.get("/ready")
def readiness_check():
    """Readiness probe that verifies model artifacts and SQLite are usable."""
    try:
        service = InferenceService.get_instance()
        db.get_analytics_summary()
        return {
            "status": "ready",
            "model_loaded": service.supervised_model is not None,
            "database_path": str(db.db_path),
        }
    except Exception as exc:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="Service dependencies are not ready") from exc


@app.post("/api/predict", response_model=PredictResponse)
def predict_risk(req: PredictRequest):
    """Real-time transaction scoring without persistent storage."""
    service = InferenceService.get_instance()
    result = service.process_transaction(req.model_dump(), persist=False)
    return result


@app.post("/api/transactions")
def submit_transaction(req: TransactionSubmitRequest):
    """Submits a transaction for scoring, SHAP explainability, and database persistence."""
    service = InferenceService.get_instance()
    result = service.process_transaction(req.model_dump(), persist=True)
    return result


@app.get("/api/transactions")
def list_transactions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    risk_band: Optional[str] = Query(None, description="Filter by LOW, MEDIUM, HIGH, or ALL"),
    search: Optional[str] = Query(None, description="Search by transaction ID, customer ID, or location"),
):
    """Retrieves paginated and filtered historical transactions."""
    items, total = db.list_transactions(limit=limit, offset=offset, risk_band=risk_band, search=search)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "transactions": items,
    }


@app.get("/api/transactions/{transaction_id}")
def get_transaction_detail(transaction_id: str):
    """Retrieves full transaction data, risk scores, and SHAP explanation payload."""
    tx = db.get_transaction(transaction_id)
    if not tx:
        raise HTTPException(status_code=404, detail=f"Transaction '{transaction_id}' not found.")
    return tx


@app.get("/api/alerts")
def get_alerts(
    status: Optional[str] = Query(None, description="Filter by NEW, UNDER_REVIEW, RESOLVED, CONFIRMED_FRAUD, or ALL"),
    limit: int = Query(50, ge=1, le=200),
):
    """Retrieves high-risk security alerts queue for analyst triage."""
    return db.list_alerts(status=status, limit=limit)


@app.patch("/api/alerts/{alert_id}")
def update_alert(alert_id: int, req: AlertUpdateRequest):
    """Updates the triage status and notes of a security alert."""
    success = db.update_alert_status(alert_id=alert_id, status=req.status, notes=req.notes)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return {"status": "success", "alert_id": alert_id, "updated_status": req.status}


@app.get("/api/analytics", response_model=AnalyticsSummary)
def get_analytics():
    """Retrieves high-level platform KPI summary and risk band distributions."""
    return db.get_analytics_summary()


@app.post("/api/seed")
def seed_database(max_records: int = 2000):
    """Seeds SQLite database from the Phase 3 scored test set."""
    test_scored_path = Path(os.getenv("FRAUD_PROCESSED_DATA_DIR", "data/processed")) / "test_scored_with_risk.csv"
    if not test_scored_path.exists():
        raise HTTPException(status_code=404, detail="Scored test set not found. Run Phase 3 first.")

    df = pd.read_csv(test_scored_path)
    db.seed_from_dataframe(df, max_records=max_records)
    summary = db.get_analytics_summary()
    return {"message": "Database successfully seeded.", "summary": summary}


# Serve Interactive Single-Page Dashboard
@app.get("/")
def serve_dashboard():
    """Serves the interactive Admin Fraud-Monitoring Dashboard."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Dashboard UI is being built. Visit /docs for the interactive API Swagger UI."}


# Mount static assets
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
