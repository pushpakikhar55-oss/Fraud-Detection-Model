"""
Data loader and ingestion pipeline for raw financial transactions.
Ensures raw data immutability and schema verification.
"""
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.data.generator import FinancialDataGenerator
from src.data.schema import validate_transaction_schema


class DataLoader:
    """Handles raw data loading, persistence, and validation."""

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def load_or_generate_raw(
        self,
        file_name: str = "transactions_raw.csv",
        num_records: int = 50000,
        num_customers: int = 1000,
        fraud_rate: float = 0.015,
        force_regenerate: bool = False,
    ) -> pd.DataFrame:
        """
        Loads raw transactions if existing, otherwise generates synthetic transactions
        and persists them immutably.
        """
        target_path = self.raw_data_dir / file_name
        
        if target_path.exists() and not force_regenerate:
            print(f"[DataLoader] Ingesting existing raw dataset from: {target_path}")
            df = pd.read_csv(target_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            print(f"[DataLoader] Generating {num_records} synthetic transactions...")
            generator = FinancialDataGenerator(
                num_customers=num_customers,
                fraud_rate=fraud_rate,
                random_seed=42,
            )
            df = generator.generate(num_records=num_records)
            self.save_raw(df, file_name=file_name)

        is_valid, errors = validate_transaction_schema(df)
        if not is_valid:
            raise ValueError(f"Raw data schema validation failed: {errors}")
            
        print(f"[DataLoader] Successfully loaded {len(df):,} transactions. Fraud rate: {df['is_fraud'].mean():.2%}")
        return df

    def save_raw(self, df: pd.DataFrame, file_name: str = "transactions_raw.csv") -> Path:
        """Saves raw transactions to immutable raw directory."""
        target_path = self.raw_data_dir / file_name
        df.to_csv(target_path, index=False)
        print(f"[DataLoader] Persisted raw dataset to: {target_path}")
        return target_path
