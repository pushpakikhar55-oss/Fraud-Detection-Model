"""
Data module: Schema definitions, synthetic data generation, and ingestion.
"""
from src.data.schema import TRANSACTION_SCHEMA, NUMERICAL_COLS, CATEGORICAL_COLS
from src.data.generator import FinancialDataGenerator
from src.data.loader import DataLoader

__all__ = [
    "TRANSACTION_SCHEMA",
    "NUMERICAL_COLS",
    "CATEGORICAL_COLS",
    "FinancialDataGenerator",
    "DataLoader",
]
