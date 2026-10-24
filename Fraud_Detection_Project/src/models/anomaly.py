"""
Unsupervised Anomaly Detection Engine (Module 6).
Implements Isolation Forest and One-Class SVM to detect zero-day fraud
and anomalous behavioral patterns without reliance on historical labels.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


class AnomalyDetectionEngine:
    """Trains unsupervised anomaly detectors and produces calibrated 0-1 anomaly risk scores."""

    def __init__(self, contamination: float = 0.02, models_dir: str = "models"):
        self.contamination = contamination
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.isolation_forest: IsolationForest = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            max_samples=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.score_min: float = -0.5
        self.score_max: float = 0.5
        self.is_fitted: bool = False
        self.metrics: Dict[str, Any] = {}

    def fit(self, X_train: pd.DataFrame) -> "AnomalyDetectionEngine":
        """Fits Isolation Forest on feature matrix."""
        print(f"[AnomalyDetectionEngine] Training Isolation Forest (contamination={self.contamination:.2%})...")
        self.isolation_forest.fit(X_train)

        # Calibrate decision_function bounds for smooth 0-1 normalization
        train_raw_scores = self.isolation_forest.decision_function(X_train)
        self.score_min = float(np.percentile(train_raw_scores, 0.5))
        self.score_max = float(np.percentile(train_raw_scores, 99.5))
        self.is_fitted = True
        return self

    def compute_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes calibrated anomaly score in range [0.0, 1.0].
        0.0 = completely normal habitual behavior
        1.0 = highly anomalous / suspicious outlier
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetectionEngine must be fitted before computing scores.")

        raw_scores = self.isolation_forest.decision_function(X)
        # raw_scores: lower values indicate outliers, higher values indicate inliers.
        # Invert and min-max normalize so higher = more anomalous.
        normalized = (self.score_max - raw_scores) / (self.score_max - self.score_min + 1e-6)
        return np.clip(normalized, 0.0, 1.0)

    def evaluate_test_set(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        test_unprocessed: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates unsupervised anomaly detection efficacy against ground truth fraud
        and breaks down detection rate across specific fraud attack vectors.
        """
        anomaly_scores = self.compute_anomaly_score(X_test)
        
        # Unsupervised prediction using contamination percentile cutoff
        threshold = float(np.percentile(anomaly_scores, 100 * (1 - self.contamination)))
        binary_preds = (anomaly_scores >= threshold).astype(int)

        roc_auc = float(roc_auc_score(y_test, anomaly_scores))
        pr_auc = float(average_precision_score(y_test, anomaly_scores))
        precision = float(precision_score(y_test, binary_preds, zero_division=0))
        recall = float(recall_score(y_test, binary_preds))
        f1 = float(f1_score(y_test, binary_preds))

        # Scenario breakdown if unprocessed metadata is provided
        scenario_capture = {}
        if test_unprocessed is not None and "fraud_scenario" in test_unprocessed.columns:
            df_eval = test_unprocessed.copy()
            df_eval["anomaly_score"] = anomaly_scores
            df_eval["anomaly_flag"] = binary_preds

            for scenario, group in df_eval[df_eval["is_fraud"] == 1].groupby("fraud_scenario"):
                detected = int(group["anomaly_flag"].sum())
                total = len(group)
                scenario_capture[scenario] = {
                    "total": total,
                    "detected": detected,
                    "recall_pct": round((detected / max(1, total)) * 100, 2),
                    "mean_anomaly_score": round(float(group["anomaly_score"].mean()), 4),
                }

        self.metrics = {
            "algorithm": "Isolation Forest",
            "contamination": self.contamination,
            "decision_threshold": round(threshold, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "scenario_capture": scenario_capture,
        }

        print(f"[AnomalyDetectionEngine] Evaluation Results:")
        print(f"  ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {recall:.2%} | F1: {f1:.4f}")
        return self.metrics

    def save(self):
        """Persists trained Isolation Forest model and metadata."""
        model_path = self.models_dir / "isolation_forest.joblib"
        joblib.dump(self, model_path)
        with open(self.models_dir / "anomaly_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[AnomalyDetectionEngine] Saved anomaly model and metrics to {model_path}")
