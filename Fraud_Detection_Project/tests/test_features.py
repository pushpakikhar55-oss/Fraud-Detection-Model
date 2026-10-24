"""
Unit tests for feature engineering logic.
"""
import pytest
import pandas as pd
from src.features.engineer import FeatureEngineer
from src.data.generator import FinancialDataGenerator


def test_feature_engineering_outputs():
    """Validates that FeatureEngineer generates all expected columns without NaNs."""
    gen = FinancialDataGenerator(num_customers=20, fraud_rate=0.05, random_seed=42)
    df = gen.generate(num_records=200, duration_days=5)

    engineer = FeatureEngineer()
    df_feat = engineer.fit_transform(df)

    expected_features = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night",
        "balance_drain_ratio",
        "balance_error_orig",
        "is_new_device",
        "is_new_location",
        "amount_to_avg_ratio",
        "amount_zscore",
        "tx_count_1h",
        "tx_count_24h",
        "tx_amount_sum_1h",
        "tx_amount_sum_24h",
        "is_high_risk_type",
        "card_not_present",
    ]

    for feat in expected_features:
        assert feat in df_feat.columns, f"Feature '{feat}' missing from output DataFrame"
        assert not df_feat[feat].isnull().any(), f"Feature '{feat}' contains null values"

    # Verify that tx_count_1h is at least 1 (since it counts itself in the rolling window)
    assert (df_feat["tx_count_1h"] >= 1).all()
