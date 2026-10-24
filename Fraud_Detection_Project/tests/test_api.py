"""
Unit and integration tests for FastAPI REST API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    """Provides a TestClient instance for testing REST endpoints."""
    return TestClient(app)


def test_health_check(client):
    """Verifies that the /health endpoint responds with 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Response-Time-Ms"]) >= 0.0


def test_readiness_check(client):
    """Verifies the readiness probe can access model artifacts and SQLite."""
    response = client.get("/ready", headers={"X-Request-ID": "phase7-readiness"})
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.headers["X-Request-ID"] == "phase7-readiness"


def test_security_headers_and_invalid_prediction_rejection(client):
    """API responses are protected against browser caching and invalid risk input."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "same-origin"
    assert response.headers["Cache-Control"] == "no-store"

    invalid = client.post("/api/predict", json={"amount": -1, "tx_count_1h": 0})
    assert invalid.status_code == 422


def test_predict_endpoint(client):
    """Tests /api/predict with low-risk and high-risk payloads."""
    # 1. Normal low-risk transaction
    normal_payload = {
        "customer_id": "CUST_TEST_01",
        "amount": 35.00,
        "transaction_type": "PAYMENT",
        "merchant_category": "retail",
        "payment_method": "credit_card",
        "old_balance": 1500.0,
        "is_night": 0,
        "is_new_device": 0,
        "is_new_location": 0,
        "tx_count_1h": 1,
        "customer_hist_mean": 45.0,
    }
    res_normal = client.post("/api/predict", json=normal_payload)
    assert res_normal.status_code == 200
    data_normal = res_normal.json()
    assert 0.0 <= data_normal["risk_score"] <= 100.0
    assert data_normal["risk_band"] in ["LOW", "MEDIUM", "HIGH"]
    assert "narrative_reasons" in data_normal

    # 2. Suspicious high-risk nocturnal transfer spike
    fraud_payload = {
        "customer_id": "CUST_TEST_02",
        "amount": 4800.00,
        "transaction_type": "TRANSFER",
        "merchant_category": "financial_services",
        "payment_method": "bank_transfer",
        "old_balance": 5000.0,
        "is_night": 1,
        "is_new_device": 1,
        "is_new_location": 1,
        "tx_count_1h": 4,
        "customer_hist_mean": 60.0,
    }
    res_fraud = client.post("/api/predict", json=fraud_payload)
    assert res_fraud.status_code == 200
    data_fraud = res_fraud.json()
    assert data_fraud["risk_score"] >= 65.0
    assert data_fraud["recommended_action"] in ["REVIEW", "BLOCK"]


def test_transaction_ingestion_and_analytics(client):
    """Tests /api/transactions persistence and /api/analytics summary."""
    tx_payload = {
        "customer_id": "CUST_API_USER",
        "amount": 120.00,
        "transaction_type": "PAYMENT",
        "merchant_category": "grocery",
        "payment_method": "debit_card",
        "old_balance": 800.0,
        "location": "Boston",
    }
    res_post = client.post("/api/transactions", json=tx_payload)
    assert res_post.status_code == 200
    tx_res = res_post.json()
    assert "transaction_id" in tx_res

    # Retrieve from transaction history
    res_list = client.get("/api/transactions?limit=10")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] >= 1

    # Check analytics endpoint
    res_analytics = client.get("/api/analytics")
    assert res_analytics.status_code == 200
    stats = res_analytics.json()
    assert stats["total_transactions"] >= 1
    assert "band_distribution" in stats
