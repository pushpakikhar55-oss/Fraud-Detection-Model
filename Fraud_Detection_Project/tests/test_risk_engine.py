"""
Unit tests for Risk Scoring Engine and Explainable AI (SHAP).
"""
import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.risk_engine.scorer import FraudRiskScorer
from src.risk_engine.explainer import FraudExplainer


def test_risk_scorer_bounds_and_bands():
    """Validates risk scoring formula bounds, component weights, and band classifications."""
    scorer = FraudRiskScorer()

    # 1. Low risk test case
    low_res = scorer.calculate_single_score(
        supervised_prob=0.01,
        anomaly_score=0.05,
        amount_to_avg_ratio=1.0,
        tx_count_1h=1,
        is_new_device=0,
        is_new_location=0,
        is_night=0,
    )
    assert 0.0 <= low_res["risk_score"] <= 30.0
    assert low_res["risk_band"] == "LOW"
    assert low_res["action"] == "APPROVE"

    # 2. High risk test case
    high_res = scorer.calculate_single_score(
        supervised_prob=0.98,
        anomaly_score=0.95,
        amount_to_avg_ratio=12.0,
        tx_count_1h=5,
        is_new_device=1,
        is_new_location=1,
        is_night=1,
    )
    assert 70.0 <= high_res["risk_score"] <= 100.0
    assert high_res["risk_band"] == "HIGH"
    assert high_res["action"] == "BLOCK"

    # 3. Batch scoring
    df_sample = pd.DataFrame({
        "amount_to_avg_ratio": [1.0, 10.0],
        "tx_count_1h": [1, 4],
        "is_new_device": [0, 1],
        "is_new_location": [0, 1],
        "is_night": [0, 1],
    })
    scored_df = scorer.score_dataframe(
        supervised_probs=np.array([0.02, 0.95]),
        anomaly_scores=np.array([0.1, 0.9]),
        df_unprocessed=df_sample,
    )
    assert "risk_score" in scored_df.columns
    assert "risk_band" in scored_df.columns
    assert "recommended_action" in scored_df.columns
    assert scored_df["risk_band"].tolist() == ["LOW", "HIGH"]


def test_explainer_local_attributions():
    """Validates that FraudExplainer produces SHAP values and human-readable narrative reasons."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "amount": np.random.uniform(10, 500, n),
        "amount_to_avg_ratio": np.random.uniform(0.5, 15.0, n),
        "tx_count_1h": np.random.randint(1, 6, n),
        "is_night": np.random.choice([0, 1], n),
        "is_new_device": np.random.choice([0, 1], n),
    })
    y = ((X["amount_to_avg_ratio"] > 5.0) & (X["is_night"] == 1)).astype(int)

    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    explainer = FraudExplainer(model=model, feature_names=list(X.columns))
    
    # Test on a simulated fraud row
    test_instance = X.iloc[0].copy()
    test_instance["amount_to_avg_ratio"] = 12.5
    test_instance["is_night"] = 1
    test_instance["amount"] = 3500.0

    explanation = explainer.explain_instance(test_instance, top_k=3)
    assert len(explanation["top_risk_drivers"]) > 0
    assert len(explanation["narrative_reasons"]) > 0
    assert any("amount" in r.lower() or "nocturnal" in r.lower() for r in explanation["narrative_reasons"])
