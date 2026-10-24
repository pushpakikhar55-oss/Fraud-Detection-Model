"""
Pydantic Request & Response Data Models for Fraud Detection APIs.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TransactionSubmitRequest(BaseModel):
    """Payload to submit a financial transaction for real-time scoring and persistence."""
    transaction_id: Optional[str] = None
    customer_id: str = Field(..., json_schema_extra={"example": "CUST_00123"})
    amount: float = Field(..., gt=0.0, json_schema_extra={"example": 450.00})
    transaction_type: str = Field(default="PAYMENT", json_schema_extra={"example": "PAYMENT"})
    merchant_category: str = Field(default="retail", json_schema_extra={"example": "retail"})
    payment_method: str = Field(default="credit_card", json_schema_extra={"example": "credit_card"})
    location: str = Field(default="New York", json_schema_extra={"example": "New York"})
    device_id: str = Field(default="DEV_10001", json_schema_extra={"example": "DEV_10001"})
    card_present: int = Field(default=1, ge=0, le=1)
    old_balance: float = Field(default=2500.0, ge=0.0)
    new_balance: Optional[float] = None
    timestamp: Optional[str] = None
    customer_hist_mean: Optional[float] = Field(default=None, json_schema_extra={"example": 85.0})


class PredictRequest(BaseModel):
    """Lightweight real-time prediction payload."""
    customer_id: str = Field(default="CUST_0001", min_length=1, max_length=128)
    amount: float = Field(default=125.00, gt=0.0)
    transaction_type: str = "PAYMENT"
    merchant_category: str = "retail"
    payment_method: str = "credit_card"
    location: str = "New York"
    device_id: str = "DEV_001"
    card_present: int = Field(default=1, ge=0, le=1)
    old_balance: float = Field(default=2000.0, ge=0.0)
    is_night: int = Field(default=0, ge=0, le=1)
    is_new_device: int = Field(default=0, ge=0, le=1)
    is_new_location: int = Field(default=0, ge=0, le=1)
    tx_count_1h: int = Field(default=1, ge=1)
    customer_hist_mean: float = Field(default=100.0, gt=0.0)


class PredictResponse(BaseModel):
    """Real-time scoring and explainability response."""
    transaction_id: str
    risk_score: float
    risk_band: str
    recommended_action: str
    supervised_probability: float
    anomaly_score: float
    narrative_reasons: List[str]
    top_risk_drivers: List[Dict[str, Any]]
    top_mitigating_factors: List[Dict[str, Any]]
    scoring_components: Dict[str, float]


class AlertUpdateRequest(BaseModel):
    """Payload to update the triage status of an alert."""
    status: str = Field(..., json_schema_extra={"example": "CONFIRMED_FRAUD"}) # RESOLVED, CONFIRMED_FRAUD, UNDER_REVIEW
    notes: Optional[str] = None


class AnalyticsSummary(BaseModel):
    """Overview analytics metrics."""
    total_transactions: int
    confirmed_fraud_count: int
    high_risk_blocked_count: int
    suspicious_review_count: int
    low_risk_approved_count: int
    active_alerts_count: int
    fraud_rate_pct: float
    loss_prevented_usd: float
    band_distribution: Dict[str, int]
    transaction_type_breakdown: List[Dict[str, Any]]
