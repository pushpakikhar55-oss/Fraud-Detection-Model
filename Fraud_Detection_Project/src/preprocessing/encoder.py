"""
Categorical encoding and numerical feature scaling transformers.
Guarantees strict separation between training set fitting and test/validation transformations.
"""
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class DataEncoder:
    """
    Fits and applies One-Hot Encoding for categorical features
    and RobustScaler for numerical features.
    """

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ):
        self.categorical_cols = categorical_cols or [
            "transaction_type",
            "merchant_category",
            "payment_method",
        ]
        self.numerical_cols = numerical_cols or [
            "amount",
            "old_balance",
            "new_balance",
            "amount_to_avg_ratio",
            "amount_zscore",
            "tx_count_1h",
            "tx_count_24h",
            "tx_amount_sum_1h",
            "tx_amount_sum_24h",
            "balance_drain_ratio",
            "balance_error_orig",
        ]
        
        # Scikit-learn transformers
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = RobustScaler()
        
        self.is_fitted: bool = False
        self.feature_names_out: List[str] = []
        self.passthrough_cols: List[str] = [
            "card_present",
            "card_not_present",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_night",
            "is_new_device",
            "is_new_location",
            "is_high_risk_type",
        ]

    def fit(self, df: pd.DataFrame) -> "DataEncoder":
        """Fits encoders strictly on training data."""
        # Fit One-Hot Encoder
        cat_data = df[self.categorical_cols].astype(str)
        self.ohe.fit(cat_data)
        encoded_cat_names = list(self.ohe.get_feature_names_out(self.categorical_cols))

        # Fit Scaler
        num_data = df[self.numerical_cols].values
        self.scaler.fit(num_data)

        # Assemble output feature names
        active_passthrough = [c for c in self.passthrough_cols if c in df.columns]
        self.feature_names_out = self.numerical_cols + encoded_cat_names + active_passthrough
        self.is_fitted = True
        print(f"[DataEncoder] Fitted encoder with {len(self.feature_names_out)} transformed features.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a DataFrame using the pre-fitted encoders."""
        if not self.is_fitted:
            raise RuntimeError("DataEncoder must be fitted before transforming data.")

        # Transform numericals
        scaled_nums = self.scaler.transform(df[self.numerical_cols].values)
        scaled_df = pd.DataFrame(scaled_nums, columns=self.numerical_cols, index=df.index)

        # Transform categoricals
        cat_data = df[self.categorical_cols].astype(str)
        encoded_cats = self.ohe.transform(cat_data)
        encoded_cat_names = list(self.ohe.get_feature_names_out(self.categorical_cols))
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoded_cat_names, index=df.index)

        # Passthrough binary/discrete features
        active_passthrough = [c for c in self.passthrough_cols if c in df.columns]
        passthrough_df = df[active_passthrough].copy()

        # Concatenate into full feature matrix
        transformed_df = pd.concat([scaled_df, encoded_cat_df, passthrough_df], axis=1)
        return transformed_df

    def save(self, filepath: str):
        """Persists fitted encoder artifact."""
        joblib.dump(self, filepath)
        print(f"[DataEncoder] Saved fitted encoder artifact to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DataEncoder":
        """Loads a persisted encoder artifact."""
        return joblib.load(filepath)
