"""
Inference and Real-Time Scoring Service.
Loads production models, scalers, and explainers; coordinates feature computation,
scoring, SHAP reasoning, and database persistence.
"""
import pathlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import os
import joblib
import numpy as np
import pandas as pd


def _load_joblib_cross_platform(path: Path):
    """
    Loads a joblib artifact regardless of which OS it was pickled on.
    Some model artifacts in this repo were saved on Windows and contain
    pickled pathlib.WindowsPath objects, which raise
    NotImplementedError on Linux/macOS (and vice versa). Temporarily
    aliasing the platform-specific Path classes during unpickling makes
    loading portable without needing to re-train or re-save the models.
    """
    posix_backup, windows_backup = pathlib.PosixPath, pathlib.WindowsPath
    try:
        pathlib.PosixPath = pathlib.PurePosixPath if os.name == "nt" else pathlib.PosixPath
        pathlib.WindowsPath = pathlib.PureWindowsPath if os.name != "nt" else pathlib.WindowsPath
        return joblib.load(path)
    finally:
        pathlib.PosixPath, pathlib.WindowsPath = posix_backup, windows_backup

from src.preprocessing.encoder import DataEncoder
from src.risk_engine.scorer import FraudRiskScorer
from src.risk_engine.explainer import FraudExplainer
from src.api.database import DatabaseManager


class InferenceService:
    """Singleton service for real-time model inference and transaction scoring."""

    _instance: Optional["InferenceService"] = None

    def __init__(self, models_dir: Optional[str] = None, data_dir: Optional[str] = None):
        self.models_dir = Path(models_dir or os.getenv("FRAUD_MODELS_DIR", "models"))
        self.data_dir = Path(data_dir or os.getenv("FRAUD_PROCESSED_DATA_DIR", "data/processed"))
        self.db = DatabaseManager()

        print("[InferenceService] Loading production artifacts into memory...")
        self.encoder = _load_joblib_cross_platform(self.data_dir / "encoder.joblib")
        self.supervised_model = _load_joblib_cross_platform(self.models_dir / "best_supervised_model.joblib")
        self.anomaly_engine = _load_joblib_cross_platform(self.models_dir / "isolation_forest.joblib")

        self.scorer = FraudRiskScorer(models_dir=str(self.models_dir))
        self.explainer = FraudExplainer(
            model=self.supervised_model,
            feature_names=self.encoder.feature_names_out,
        )
        print("[InferenceService] All production models and transformers successfully initialized.")

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_transaction(self, tx_dict: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
        """
        Executes end-to-end real-time scoring on an incoming transaction:
        1. Feature derivation
        2. Scaler/Encoder transformation
        3. Supervised & Anomaly inferences
        4. Multi-signal 0-100 risk scoring
        5. Local SHAP narrative explanations
        6. DB persistence and auto-alerting
        """
        now = datetime.now(timezone.utc)
        tx_id = tx_dict.get("transaction_id") or f"TXN_{int(now.timestamp()*1000)}"
        timestamp_val = tx_dict.get("timestamp") or now.isoformat()
        dt = pd.to_datetime(timestamp_val)

        amount = float(tx_dict.get("amount", 100.0))
        customer_id = str(tx_dict.get("customer_id", "CUST_DEFAULT"))
        old_bal = float(tx_dict.get("old_balance", 2000.0))
        tx_type = str(tx_dict.get("transaction_type", "PAYMENT"))
        
        # New balance calculation if not supplied
        if "new_balance" in tx_dict and tx_dict["new_balance"] is not None:
            new_bal = float(tx_dict["new_balance"])
        else:
            new_bal = max(0.0, old_bal - amount) if tx_type in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"] else old_bal + amount

        # Behavioral and temporal feature calculation
        hist_mean = float(tx_dict.get("customer_hist_mean") or 100.0)
        amount_to_avg_ratio = amount / max(1.0, hist_mean)
        amount_zscore = (amount - hist_mean) / max(1.0, hist_mean * 0.4)

        hour = dt.hour
        day_of_week = dt.dayofweek
        is_weekend = int(day_of_week in [5, 6])
        is_night = int(hour in [23, 0, 1, 2, 3, 4, 5]) if "is_night" not in tx_dict else int(tx_dict["is_night"])
        is_new_device = int(tx_dict.get("is_new_device", 0))
        is_new_location = int(tx_dict.get("is_new_location", 0))
        tx_count_1h = int(tx_dict.get("tx_count_1h", 1))
        tx_amount_sum_1h = amount * tx_count_1h
        tx_count_24h = max(tx_count_1h, int(tx_dict.get("tx_count_24h", tx_count_1h)))
        tx_amount_sum_24h = tx_amount_sum_1h

        balance_drain_ratio = float(np.clip(amount / (old_bal + 1e-5), 0.0, 5.0)) if old_bal > 0 else 1.0
        expected_bal = max(0.0, old_bal - amount) if tx_type in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"] else old_bal + amount
        balance_error_orig = float(abs(new_bal - expected_bal))

        card_present = int(tx_dict.get("card_present", 1))
        card_not_present = 1 if card_present == 0 else 0
        is_high_risk_type = int(tx_type in ["TRANSFER", "CASH_OUT"])

        # Construct raw row DataFrame matching encoder expectation
        raw_row = {
            "transaction_id": tx_id,
            "timestamp": dt,
            "customer_id": customer_id,
            "amount": amount,
            "transaction_type": tx_type,
            "merchant_id": tx_dict.get("merchant_id", "MERC_0001"),
            "merchant_category": tx_dict.get("merchant_category", "retail"),
            "device_id": tx_dict.get("device_id", "DEV_10001"),
            "location": tx_dict.get("location", "New York"),
            "payment_method": tx_dict.get("payment_method", "credit_card"),
            "card_present": card_present,
            "card_not_present": card_not_present,
            "old_balance": old_bal,
            "new_balance": new_bal,
            "amount_to_avg_ratio": amount_to_avg_ratio,
            "amount_zscore": amount_zscore,
            "tx_count_1h": tx_count_1h,
            "tx_count_24h": tx_count_24h,
            "tx_amount_sum_1h": tx_amount_sum_1h,
            "tx_amount_sum_24h": tx_amount_sum_24h,
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "is_new_device": is_new_device,
            "is_new_location": is_new_location,
            "balance_drain_ratio": balance_drain_ratio,
            "balance_error_orig": balance_error_orig,
            "is_high_risk_type": is_high_risk_type,
            "customer_hist_mean": hist_mean,
        }
        df_raw = pd.DataFrame([raw_row])

        # Transform features
        transformed_X = self.encoder.transform(df_raw)

        # Model Inferences
        supervised_prob = float(self.supervised_model.predict_proba(transformed_X)[0, 1])
        anomaly_score = float(self.anomaly_engine.compute_anomaly_score(transformed_X)[0])

        # Composite Risk Scoring
        score_res = self.scorer.calculate_single_score(
            supervised_prob=supervised_prob,
            anomaly_score=anomaly_score,
            amount_to_avg_ratio=amount_to_avg_ratio,
            tx_count_1h=tx_count_1h,
            is_new_device=is_new_device,
            is_new_location=is_new_location,
            is_night=is_night,
        )

        # SHAP Explainability & Narrative Reason Synthesis
        explanation = self.explainer.explain_instance(
            instance_features=transformed_X.iloc[0],
            raw_row=pd.Series(raw_row),
            top_k=5,
        )

        result = {
            "transaction_id": tx_id,
            "timestamp": dt.isoformat(),
            "customer_id": customer_id,
            "amount": amount,
            "transaction_type": tx_type,
            "merchant_category": raw_row["merchant_category"],
            "payment_method": raw_row["payment_method"],
            "location": raw_row["location"],
            "device_id": raw_row["device_id"],
            "card_present": card_present,
            "old_balance": old_bal,
            "new_balance": new_bal,
            "risk_score": score_res["risk_score"],
            "risk_band": score_res["risk_band"],
            "recommended_action": score_res["action"],
            "supervised_probability": score_res["signals"]["supervised_probability"],
            "anomaly_score": score_res["signals"]["anomaly_score"],
            "scoring_components": score_res["components"],
            "narrative_reasons": explanation["narrative_reasons"],
            "top_risk_drivers": explanation["top_risk_drivers"],
            "top_mitigating_factors": explanation["top_mitigating_factors"],
        }

        if persist:
            db_record = {
                **raw_row,
                "risk_score": result["risk_score"],
                "risk_band": result["risk_band"],
                "recommended_action": result["recommended_action"],
                "supervised_prob": result["supervised_probability"],
                "anomaly_score": result["anomaly_score"],
                "explanation_json": {
                    "narrative_reasons": result["narrative_reasons"],
                    "top_risk_drivers": result["top_risk_drivers"],
                    "top_mitigating_factors": result["top_mitigating_factors"],
                    "scoring_components": result["scoring_components"],
                }
            }
            self.db.insert_transaction(db_record)

        return result
