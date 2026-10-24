"""
Transaction schema, column constants, data types, and validation logic.
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Canonical Transaction Schema
TRANSACTION_SCHEMA: Dict[str, str] = {
    "transaction_id": "string",
    "timestamp": "datetime64[ns]",
    "customer_id": "string",
    "amount": "float64",
    "transaction_type": "string",
    "merchant_id": "string",
    "merchant_category": "string",
    "device_id": "string",
    "location": "string",
    "payment_method": "string",
    "card_present": "int64",
    "old_balance": "float64",
    "new_balance": "float64",
    "is_fraud": "int64",
    "fraud_scenario": "string",
}

TRANSACTION_TYPES: List[str] = [
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "DEBIT",
    "CASH_IN",
]

MERCHANT_CATEGORIES: List[str] = [
    "retail",
    "travel",
    "digital_goods",
    "entertainment",
    "grocery",
    "food_dining",
    "utilities",
    "financial_services",
]

PAYMENT_METHODS: List[str] = [
    "credit_card",
    "debit_card",
    "bank_transfer",
    "upi",
    "digital_wallet",
]

POPULAR_LOCATIONS: List[str] = [
    "New York",
    "San Francisco",
    "Chicago",
    "Houston",
    "Miami",
    "Seattle",
    "Boston",
    "Austin",
    "Denver",
    "Atlanta",
    "International",
]

# Column groupings for ML Pipelines
IDENTIFIER_COLS: List[str] = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
]

TARGET_COL: str = "is_fraud"
EXPLANATION_COL: str = "fraud_scenario"
DATETIME_COL: str = "timestamp"

NUMERICAL_COLS: List[str] = [
    "amount",
    "old_balance",
    "new_balance",
]

CATEGORICAL_COLS: List[str] = [
    "transaction_type",
    "merchant_category",
    "payment_method",
    "location",
]

BINARY_COLS: List[str] = [
    "card_present",
]


def validate_transaction_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame conforms to the expected canonical transaction schema.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    # Check for missing required columns
    required_cols = set(TRANSACTION_SCHEMA.keys())
    present_cols = set(df.columns)
    missing = required_cols - present_cols
    if missing:
        errors.append(f"Missing required columns: {sorted(list(missing))}")
        return False, errors
    
    # Check data consistency
    if (df["amount"] < 0).any():
        errors.append("Dataset contains negative amounts.")
    if (df["old_balance"] < 0).any():
        errors.append("Dataset contains negative old_balance values.")
    if (df["new_balance"] < 0).any():
        errors.append("Dataset contains negative new_balance values.")
    if not set(df["is_fraud"].unique()).issubset({0, 1}):
        errors.append("Target 'is_fraud' must contain only binary values (0 or 1).")
        
    is_valid = len(errors) == 0
    return is_valid, errors
