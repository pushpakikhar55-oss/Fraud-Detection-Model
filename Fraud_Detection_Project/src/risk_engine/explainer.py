"""
Explainable AI Engine (Module 8).
Utilizes SHAP (SHapley Additive exPlanations) to provide local feature attributions,
generates natural-language reasoning narratives, and renders visual explainability artifacts.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import shap
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class FraudExplainer:
    """Provides local and global feature attribution explanations using SHAP."""

    def __init__(self, model: Optional[Any] = None, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None

        if self.model is not None:
            self._init_explainer()

    def _init_explainer(self):
        """Initializes SHAP TreeExplainer for tree models."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print("[FraudExplainer] Initialized SHAP TreeExplainer successfully.")
        except Exception as e:
            print(f"[FraudExplainer] Warning: Could not initialize TreeExplainer: {e}. Falling back to KernelExplainer.")
            self.explainer = None

    def explain_instance(
        self,
        instance_features: Union[pd.Series, pd.DataFrame, np.ndarray],
        raw_row: Optional[pd.Series] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Computes local Shapley attributions and synthesizes human-readable narrative explanations.
        """
        if self.explainer is None:
            raise RuntimeError("FraudExplainer must be initialized with a fitted model.")

        if isinstance(instance_features, pd.Series):
            df_inst = pd.DataFrame([instance_features])
        elif isinstance(instance_features, np.ndarray):
            df_inst = pd.DataFrame(instance_features.reshape(1, -1), columns=self.feature_names)
        else:
            df_inst = instance_features.copy()

        # Compute SHAP values
        shap_values = self.explainer.shap_values(df_inst)
        if isinstance(shap_values, list):
            # Binary classification list format [class_0, class_1]
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values

        feature_names = list(df_inst.columns)
        contributions = []
        for name, val, shap_val in zip(feature_names, df_inst.iloc[0].values, sv):
            contributions.append({
                "feature": name,
                "feature_value": float(val),
                "shap_value": float(shap_val),
                "absolute_impact": abs(float(shap_val)),
            })

        # Sort by impact
        contributions.sort(key=lambda x: x["absolute_impact"], reverse=True)
        top_positive = [c for c in contributions if c["shap_value"] > 0][:top_k]
        top_negative = [c for c in contributions if c["shap_value"] < 0][:top_k]

        # Generate human-readable narrative reasons
        narrative_reasons = self._generate_narrative_reasons(df_inst.iloc[0], top_positive, raw_row)

        return {
            "top_risk_drivers": top_positive,
            "top_mitigating_factors": top_negative,
            "narrative_reasons": narrative_reasons,
            "base_value": float(self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) else self.explainer.expected_value[1]),
        }

    def _generate_narrative_reasons(
        self,
        instance: pd.Series,
        top_positive: List[Dict[str, Any]],
        raw_row: Optional[pd.Series] = None,
    ) -> List[str]:
        """Translates top mathematical SHAP risk drivers into plain English explanations."""
        reasons = []
        top_feature_names = {item["feature"] for item in top_positive}

        # Check raw row context if available, otherwise check transformed instance
        row = raw_row if raw_row is not None else instance

        # Rule 1: Amount spike
        if ("amount_to_avg_ratio" in top_feature_names or "amount" in top_feature_names) and "amount_to_avg_ratio" in row:
            ratio = float(row["amount_to_avg_ratio"])
            if ratio >= 2.5:
                amt = float(row.get("amount", 0.0))
                reasons.append(f"Unusually high transaction amount (${amt:,.2f}), which is {ratio:.1f}x higher than customer's historical average.")

        # Rule 2: Velocity burst
        if ("tx_count_1h" in top_feature_names or "tx_amount_sum_1h" in top_feature_names) and "tx_count_1h" in row:
            cnt = int(row["tx_count_1h"])
            if cnt >= 2:
                reasons.append(f"Elevated velocity burst: {cnt} transactions initiated within the past 1 hour.")

        # Rule 3: Nocturnal execution
        if "is_night" in top_feature_names and int(row.get("is_night", 0)) == 1:
            hr = int(row.get("hour_of_day", 2))
            reasons.append(f"High-risk execution window: Transaction initiated at {hr:02d}:00 during nocturnal hours.")

        # Rule 4: New device fingerprint
        if "is_new_device" in top_feature_names and int(row.get("is_new_device", 0)) == 1:
            dev_id = str(row.get("device_id", "unrecognized"))
            reasons.append(f"Unrecognized hardware fingerprint: First recorded access from device {dev_id}.")

        # Rule 5: New geographic location
        if "is_new_location" in top_feature_names and int(row.get("is_new_location", 0)) == 1:
            loc = str(row.get("location", "anomalous region"))
            reasons.append(f"Geographic novelty: Initiated from previously unvisited location ({loc}).")

        # Rule 6: Liquidation / Balance Drain
        if "balance_drain_ratio" in top_feature_names and "balance_drain_ratio" in row:
            drain = float(row["balance_drain_ratio"])
            if drain >= 0.85:
                reasons.append(f"Severe account liquidation: Transaction drains {drain*100:.1f}% of entire existing account balance.")

        # Rule 7: High-risk exit transfer
        if "is_high_risk_type" in top_feature_names and int(row.get("is_high_risk_type", 0)) == 1:
            tx_type = str(row.get("transaction_type", "TRANSFER / CASH_OUT"))
            reasons.append(f"Irreversible exit channel utilized: Transaction executed via {tx_type}.")

        if not reasons:
            reasons.append("Multiple low-level behavioral and contextual signals converged to exceed risk thresholds.")

        return reasons

    def generate_visual_explanations(
        self,
        X_test: pd.DataFrame,
        sample_fraud_idx: int,
        figures_dir: str = "reports/figures",
    ):
        """Generates global SHAP summary plot and local waterfall plot."""
        fig_dir = Path(figures_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        if self.explainer is None:
            return

        print("[FraudExplainer] Calculating SHAP values for global summary plot (sample of 500 test points)...")
        # Subsample for fast rendering
        sample_X = X_test.sample(n=min(500, len(X_test)), random_state=42)
        shap_values_sample = self.explainer(sample_X)

        # 1. Global Beeswarm Summary Plot
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_values_sample, max_display=12, show=False)
        plt.title("SHAP Global Feature Importance (Beeswarm Attribution)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(fig_dir / "shap_global_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Local Waterfall Plot for a Flagged Fraud Instance
        plt.figure(figsize=(9, 6))
        sample_explanation = self.explainer(X_test.iloc[[sample_fraud_idx]])
        shap.plots.waterfall(sample_explanation[0], max_display=10, show=False)
        plt.title("SHAP Local Attribution Waterfall (Flagged Fraud Case)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(fig_dir / "shap_waterfall_example.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[FraudExplainer] Saved SHAP visual explanations to {fig_dir}/")
