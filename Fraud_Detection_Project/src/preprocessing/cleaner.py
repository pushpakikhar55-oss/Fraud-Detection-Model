"""
Data cleaning and quality assurance module.
Handles deduplication, invalid records, null imputation, and data quality reporting.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data.schema import NUMERICAL_COLS, CATEGORICAL_COLS


class DataCleaner:
    """Cleans raw transaction data and reports data quality health metrics."""

    def __init__(self):
        self.quality_report: Dict[str, Any] = {}

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes end-to-end cleaning operations and compiles a quality report.
        """
        initial_count = len(df)
        df_clean = df.copy()

        # 1. Duplication audit
        duplicate_mask = df_clean.duplicated(subset=["transaction_id"], keep="first")
        num_duplicates = int(duplicate_mask.sum())
        if num_duplicates > 0:
            df_clean = df_clean[~duplicate_mask]

        # 2. Null audit & imputation
        null_counts = df_clean.isnull().sum().to_dict()
        
        # Impute numericals with median
        for col in NUMERICAL_COLS:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                
        # Impute categoricals with 'UNKNOWN'
        for col in CATEGORICAL_COLS:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna("UNKNOWN")

        # 3. Invalid record filtering
        valid_mask = (
            (df_clean["amount"] > 0)
            & (df_clean["old_balance"] >= 0)
            & (df_clean["new_balance"] >= 0)
            & (df_clean["timestamp"].notnull())
        )
        invalid_count = int((~valid_mask).sum())
        df_clean = df_clean[valid_mask]

        final_count = len(df_clean)

        self.quality_report = {
            "initial_records": initial_count,
            "final_records": final_count,
            "duplicates_removed": num_duplicates,
            "invalid_records_removed": invalid_count,
            "null_values_imputed": {k: int(v) for k, v in null_counts.items() if v > 0},
            "retention_rate_pct": round((final_count / max(1, initial_count)) * 100, 2),
        }

        print(f"[DataCleaner] Retained {final_count:,} records ({self.quality_report['retention_rate_pct']}%)")
        return df_clean, self.quality_report
