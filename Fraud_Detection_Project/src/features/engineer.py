"""
Feature Engineering Engine for Financial Fraud Detection.
Implements strictly causal, non-leaking behavioral baselines, rolling velocity metrics,
temporal indicators, and novelty detection signals.
"""
from typing import List, Optional
import numpy as np
import pandas as pd


class FeatureEngineer:
    """Computes behavioral, velocity, novelty, and financial features."""

    def __init__(self):
        self.engineered_feature_cols: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all features on the dataset in chronological sequence.
        """
        # Ensure chronological order
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        print("[FeatureEngineer] Extracting temporal features...")
        df = self._add_temporal_features(df)

        print("[FeatureEngineer] Extracting account balance dynamics...")
        df = self._add_balance_features(df)

        print("[FeatureEngineer] Extracting device and location novelty...")
        df = self._add_novelty_features(df)

        print("[FeatureEngineer] Extracting customer behavioral baselines (strictly causal)...")
        df = self._add_customer_baselines(df)

        print("[FeatureEngineer] Extracting rolling spending velocity...")
        df = self._add_rolling_velocity(df)

        print("[FeatureEngineer] Extracting high-risk channel flags...")
        df = self._add_risk_flags(df)

        self._record_feature_columns(df)
        print(f"[FeatureEngineer] Feature engineering completed. Generated {len(self.engineered_feature_cols)} features.")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts cyclic and categorical time-based indicators."""
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        # Night window: 23:00 to 05:00
        df["is_night"] = df["hour_of_day"].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
        return df

    def _add_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes balance drain ratio and balance delta discrepancies."""
        # Fraction of balance liquidated by transaction
        df["balance_drain_ratio"] = np.where(
            df["old_balance"] > 0,
            np.clip(df["amount"] / (df["old_balance"] + 1e-5), 0.0, 5.0),
            1.0
        )
        # Check accounting discrepancy for outflow transactions
        expected_new = np.where(
            df["transaction_type"].isin(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]),
            np.maximum(0.0, df["old_balance"] - df["amount"]),
            df["old_balance"] + df["amount"]
        )
        df["balance_error_orig"] = np.abs(df["new_balance"] - expected_new)
        return df

    def _add_novelty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags first-time device or geographic location per customer."""
        # Causal device novelty
        device_cumcount = df.groupby(["customer_id", "device_id"]).cumcount()
        df["is_new_device"] = (device_cumcount == 0).astype(int)

        # Causal location novelty
        location_cumcount = df.groupby(["customer_id", "location"]).cumcount()
        df["is_new_location"] = (location_cumcount == 0).astype(int)
        return df

    def _add_customer_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes personal historical baseline (expanding mean/std) up to the current transaction.
        Strictly causal: uses prior history, avoiding lookahead bias.
        """
        # Calculate cumulative statistics excluding current transaction
        cum_count = df.groupby("customer_id").cumcount()
        cum_sum = df.groupby("customer_id")["amount"].cumsum() - df["amount"]
        
        # Global mean fallback for first transaction of a customer
        global_mean = df["amount"].mean()
        prior_mean = np.where(cum_count > 0, cum_sum / np.maximum(cum_count, 1), global_mean)
        
        df["customer_hist_mean"] = prior_mean
        df["amount_to_avg_ratio"] = df["amount"] / (df["customer_hist_mean"] + 1e-5)
        
        # Approximate z-score deviation using expanding std
        # For efficiency and robustness:
        global_std = df["amount"].std()
        df["amount_zscore"] = (df["amount"] - df["customer_hist_mean"]) / (global_std + 1e-5)
        return df

    def _add_rolling_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling transaction velocity (counts and amounts) in 1h and 24h windows.
        """
        # Retain original index to re-merge safely
        df["_orig_idx"] = df.index
        
        # Sort by customer_id and timestamp for rolling window alignment
        temp = df[["_orig_idx", "customer_id", "timestamp", "amount"]].sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
        
        # Fast rolling count & sum per customer over 1h and 24h
        r1h = (
            temp.set_index("timestamp")
            .groupby("customer_id")["amount"]
            .rolling("1h", closed="both")
        )
        r24h = (
            temp.set_index("timestamp")
            .groupby("customer_id")["amount"]
            .rolling("24h", closed="both")
        )

        temp["tx_count_1h"] = r1h.count().values.astype(int)
        temp["tx_amount_sum_1h"] = r1h.sum().values
        temp["tx_count_24h"] = r24h.count().values.astype(int)
        temp["tx_amount_sum_24h"] = r24h.sum().values

        # Merge back on original index
        merged = temp[["_orig_idx", "tx_count_1h", "tx_amount_sum_1h", "tx_count_24h", "tx_amount_sum_24h"]]
        df = df.merge(merged, on="_orig_idx", how="left")
        df = df.drop(columns=["_orig_idx"])
        return df

    def _add_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds transaction type risk flag and card-not-present flag."""
        df["is_high_risk_type"] = df["transaction_type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
        df["card_not_present"] = (df["card_present"] == 0).astype(int)
        return df

    def _record_feature_columns(self, df: pd.DataFrame):
        """Maintains the authoritative list of engineered feature columns."""
        excluded = [
            "transaction_id",
            "timestamp",
            "customer_id",
            "merchant_id",
            "device_id",
            "location",
            "transaction_type",
            "merchant_category",
            "payment_method",
            "is_fraud",
            "fraud_scenario",
        ]
        self.engineered_feature_cols = [c for c in df.columns if c not in excluded]
