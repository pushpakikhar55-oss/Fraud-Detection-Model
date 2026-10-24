"""
Unit tests for data generator and schema validation.
"""
import pytest
import pandas as pd
from src.data.generator import FinancialDataGenerator
from src.data.schema import validate_transaction_schema, TRANSACTION_SCHEMA


def test_generator_schema_validity():
    """Validates that generated transactions strictly match canonical schema."""
    gen = FinancialDataGenerator(num_customers=50, fraud_rate=0.05, random_seed=42)
    df = gen.generate(num_records=500, duration_days=10)
    
    assert len(df) == 500
    is_valid, errors = validate_transaction_schema(df)
    assert is_valid, f"Schema validation errors: {errors}"


def test_fraud_scenarios_presence():
    """Verifies that injected fraud scenarios are present in positive fraud records."""
    gen = FinancialDataGenerator(num_customers=50, fraud_rate=0.10, random_seed=42)
    df = gen.generate(num_records=500, duration_days=10)
    
    fraud_df = df[df["is_fraud"] == 1]
    assert len(fraud_df) > 0
    scenarios = set(fraud_df["fraud_scenario"].unique())
    assert len(scenarios) > 1, f"Expected multiple fraud scenarios, found: {scenarios}"
    
    # Legit records should only have scenario 'none'
    legit_df = df[df["is_fraud"] == 0]
    assert (legit_df["fraud_scenario"] == "none").all()
