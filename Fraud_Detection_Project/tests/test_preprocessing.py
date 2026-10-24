"""
Unit tests for data cleaning, encoding, and leakage prevention.
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import DataEncoder


def test_cleaner_removes_duplicates_and_invalids():
    """Tests that duplicates, negative amounts, and missing values are handled cleanly."""
    raw_data = pd.DataFrame({
        "transaction_id": ["T1", "T1", "T2", "T3"],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"]),
        "customer_id": ["C1", "C1", "C2", "C3"],
        "amount": [100.0, 100.0, -50.0, np.nan],  # T1 dup, T2 negative, T3 missing
        "old_balance": [500.0, 500.0, 200.0, 300.0],
        "new_balance": [400.0, 400.0, 250.0, 300.0],
        "transaction_type": ["PAYMENT", "PAYMENT", "TRANSFER", np.nan],
        "merchant_category": ["retail", "retail", "travel", "grocery"],
        "payment_method": ["credit_card", "credit_card", "upi", "debit_card"],
        "is_fraud": [0, 0, 1, 0],
    })

    cleaner = DataCleaner()
    df_clean, report = cleaner.clean(raw_data)

    # T1 duplicate should be dropped (1 removed)
    # T2 negative amount should be dropped (1 removed)
    # T3 missing amount is imputed with median and retained
    assert len(df_clean) == 2
    assert "T1" in df_clean["transaction_id"].values
    assert "T2" not in df_clean["transaction_id"].values
    assert report["duplicates_removed"] == 1
    assert report["invalid_records_removed"] == 1


def test_encoder_leakage_free():
    """Ensures encoder fits on training data and transforms test data cleanly without leakage."""
    train_data = pd.DataFrame({
        "amount": [10.0, 20.0, 30.0],
        "old_balance": [100.0, 200.0, 300.0],
        "new_balance": [90.0, 180.0, 270.0],
        "amount_to_avg_ratio": [1.0, 1.0, 1.0],
        "amount_zscore": [0.0, 0.0, 0.0],
        "tx_count_1h": [1, 1, 1],
        "tx_count_24h": [1, 1, 1],
        "tx_amount_sum_1h": [10.0, 20.0, 30.0],
        "tx_amount_sum_24h": [10.0, 20.0, 30.0],
        "balance_drain_ratio": [0.1, 0.1, 0.1],
        "balance_error_orig": [0.0, 0.0, 0.0],
        "transaction_type": ["PAYMENT", "TRANSFER", "PAYMENT"],
        "merchant_category": ["retail", "travel", "retail"],
        "payment_method": ["credit_card", "upi", "credit_card"],
        "card_present": [1, 0, 1],
        "card_not_present": [0, 1, 0],
        "hour_of_day": [12, 14, 16],
        "day_of_week": [0, 1, 2],
        "is_weekend": [0, 0, 0],
        "is_night": [0, 0, 0],
        "is_new_device": [1, 0, 0],
        "is_new_location": [1, 0, 0],
        "is_high_risk_type": [0, 1, 0],
    })

    test_data = train_data.copy()
    test_data["amount"] = [1000.0, 2000.0, 3000.0]  # extreme test values

    encoder = DataEncoder()
    encoder.fit(train_data)
    
    transformed_train = encoder.transform(train_data)
    transformed_test = encoder.transform(test_data)

    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert not transformed_train.isnull().values.any()
    assert not transformed_test.isnull().values.any()
