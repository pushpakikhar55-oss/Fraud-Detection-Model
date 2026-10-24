"""
Hybrid 0-100 Fraud Risk Scoring Engine.
Combines supervised probability, unsupervised anomaly scores, behavioral deviation,
spending velocity, and novelty markers into calibrated operational risk bands.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union
import numpy as np
import pandas as pd


class FraudRiskScorer:
    """Computes calibrated 0-100 risk scores and assigns policy decision recommendations."""

    # Canonical Risk Bands & Automated Policy Triggers
    BANDS: Dict[str, Dict[str, Any]] = {
        "LOW": {
            "min_score": 0.0,
            "max_score": 30.0,
            "action": "APPROVE",
            "description": "Legitimate transaction pattern. Low risk of fraudulent activity.",
            "friction": "None (Seamless authorization)",
        },
        "MEDIUM": {
            "min_score": 30.01,
            "max_score": 70.0,
            "action": "REVIEW",
            "description": "Suspicious deviation detected. Warranting secondary authorization.",
            "friction": "Step-up 2FA / OTP challenge or analyst queue",
        },
        "HIGH": {
            "min_score": 70.01,
            "max_score": 100.0,
            "action": "BLOCK",
            "description": "Critical fraud indicators confirmed. High financial loss exposure.",
            "friction": "Immediate transaction decline & security alert",
        },
    }

    def __init__(
        self,
        weight_supervised: float = 0.40,
        weight_anomaly: float = 0.25,
        weight_amount_dev: float = 0.15,
        weight_velocity: float = 0.10,
        weight_novelty: float = 0.10,
        models_dir: str = "models",
    ):
        self.w_sup = weight_supervised
        self.w_anom = weight_anomaly
        self.w_amt = weight_amount_dev
        self.w_vel = weight_velocity
        self.w_nov = weight_novelty
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def calculate_single_score(
        self,
        supervised_prob: float,
        anomaly_score: float,
        amount_to_avg_ratio: float = 1.0,
        tx_count_1h: int = 1,
        is_new_device: int = 0,
        is_new_location: int = 0,
        is_night: int = 0,
    ) -> Dict[str, Any]:
        """Calculates multi-signal risk score for a single transaction event."""
        # 1. Normalized Amount Deviation sub-score [0, 1]
        amt_subscore = float(np.clip((amount_to_avg_ratio - 1.0) / 8.0, 0.0, 1.0))

        # 2. Normalized Velocity sub-score [0, 1]
        vel_subscore = float(np.clip((tx_count_1h - 1) / 4.0, 0.0, 1.0))

        # 3. Contextual Novelty sub-score [0, 1]
        nov_subscore = float(0.4 * is_new_device + 0.3 * is_new_location + 0.3 * is_night)

        # Composite weighted 0-100 formula
        raw_score = (
            (self.w_sup * supervised_prob)
            + (self.w_anom * anomaly_score)
            + (self.w_amt * amt_subscore)
            + (self.w_vel * vel_subscore)
            + (self.w_nov * nov_subscore)
        ) * 100.0

        risk_score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)
        band_name, band_info = self.classify_band(risk_score)

        return {
            "risk_score": risk_score,
            "risk_band": band_name,
            "action": band_info["action"],
            "components": {
                "supervised_component": round(self.w_sup * supervised_prob * 100, 1),
                "anomaly_component": round(self.w_anom * anomaly_score * 100, 1),
                "amount_deviation_component": round(self.w_amt * amt_subscore * 100, 1),
                "velocity_component": round(self.w_vel * vel_subscore * 100, 1),
                "novelty_component": round(self.w_nov * nov_subscore * 100, 1),
            },
            "signals": {
                "supervised_probability": round(supervised_prob, 4),
                "anomaly_score": round(anomaly_score, 4),
                "amount_to_avg_ratio": round(amount_to_avg_ratio, 2),
                "tx_count_1h": int(tx_count_1h),
                "is_new_device": int(is_new_device),
                "is_new_location": int(is_new_location),
                "is_night": int(is_night),
            }
        }

    def score_dataframe(
        self,
        supervised_probs: np.ndarray,
        anomaly_scores: np.ndarray,
        df_unprocessed: pd.DataFrame,
    ) -> pd.DataFrame:
        """Computes risk scores in bulk for a dataframe of transactions."""
        n = len(supervised_probs)
        amt_ratios = df_unprocessed["amount_to_avg_ratio"].values if "amount_to_avg_ratio" in df_unprocessed.columns else np.ones(n)
        tx_counts = df_unprocessed["tx_count_1h"].values if "tx_count_1h" in df_unprocessed.columns else np.ones(n)
        new_devs = df_unprocessed["is_new_device"].values if "is_new_device" in df_unprocessed.columns else np.zeros(n)
        new_locs = df_unprocessed["is_new_location"].values if "is_new_location" in df_unprocessed.columns else np.zeros(n)
        nights = df_unprocessed["is_night"].values if "is_night" in df_unprocessed.columns else np.zeros(n)

        # Vectorized sub-scores
        amt_sub = np.clip((amt_ratios - 1.0) / 8.0, 0.0, 1.0)
        vel_sub = np.clip((tx_counts - 1.0) / 4.0, 0.0, 1.0)
        nov_sub = 0.4 * new_devs + 0.3 * new_locs + 0.3 * nights

        raw_scores = (
            (self.w_sup * supervised_probs)
            + (self.w_anom * anomaly_scores)
            + (self.w_amt * amt_sub)
            + (self.w_vel * vel_sub)
            + (self.w_nov * nov_sub)
        ) * 100.0

        risk_scores = np.round(np.clip(raw_scores, 0.0, 100.0), 1)

        # Assign bands
        bands = []
        actions = []
        for s in risk_scores:
            b, info = self.classify_band(s)
            bands.append(b)
            actions.append(info["action"])

        df_out = df_unprocessed.copy()
        df_out["risk_score"] = risk_scores
        df_out["risk_band"] = bands
        df_out["recommended_action"] = actions
        df_out["supervised_prob"] = np.round(supervised_probs, 4)
        df_out["anomaly_score"] = np.round(anomaly_scores, 4)
        return df_out

    @classmethod
    def classify_band(cls, risk_score: float) -> Tuple[str, Dict[str, Any]]:
        """Categorizes a continuous 0-100 score into Low, Medium, or High risk bands."""
        if risk_score <= 30.0:
            return "LOW", cls.BANDS["LOW"]
        elif risk_score <= 70.0:
            return "MEDIUM", cls.BANDS["MEDIUM"]
        else:
            return "HIGH", cls.BANDS["HIGH"]

    def save_configuration(self):
        """Persists risk scoring weights and band configurations."""
        config = {
            "weights": {
                "supervised": self.w_sup,
                "anomaly": self.w_anom,
                "amount_deviation": self.w_amt,
                "velocity": self.w_vel,
                "novelty": self.w_nov,
            },
            "bands": self.BANDS,
        }
        with open(self.models_dir / "risk_scorer_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"[FraudRiskScorer] Saved configuration to {self.models_dir / 'risk_scorer_config.json'}")
