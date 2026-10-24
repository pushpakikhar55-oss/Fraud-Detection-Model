"""
End-to-end Preprocessing Pipeline and Dataset Partitioning Manager.
Coordinates cleaning, feature engineering, temporal/stratified splitting,
encoding, and class imbalance parameter calculation.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.preprocessing.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer
from src.preprocessing.encoder import DataEncoder
from src.data.schema import TARGET_COL, EXPLANATION_COL


class PreprocessingPipeline:
    """Orchestrates end-to-end data preparation for modeling."""

    def __init__(
        self,
        output_dir: str = "data/processed",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.encoder = DataEncoder()

    def process(
        self,
        df_raw: pd.DataFrame,
        split_method: str = "temporal",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes end-to-end preprocessing:
        1. Clean raw data
        2. Feature engineering
        3. Split into Train / Val / Test (temporal or stratified)
        4. Fit encoder on Train ONLY and transform Val and Test
        5. Persist processed datasets and compute class imbalance parameters
        """
        print("\n" + "="*60)
        print("STAGE 1: CLEANING RAW DATASET")
        print("="*60)
        df_clean, clean_report = self.cleaner.clean(df_raw)

        print("\n" + "="*60)
        print("STAGE 2: FEATURE ENGINEERING")
        print("="*60)
        df_featured = self.engineer.fit_transform(df_clean)

        print("\n" + "="*60)
        print(f"STAGE 3: DATASET SPLITTING (Strategy: {split_method.upper()})")
        print("="*60)
        train_df, val_df, test_df = self._split_data(df_featured, method=split_method)

        print("\n" + "="*60)
        print("STAGE 4: CATEGORICAL ENCODING & NUMERICAL SCALING")
        print("="*60)
        # Fit encoder on Train ONLY
        self.encoder.fit(train_df)
        
        # Transform all three splits
        X_train = self.encoder.transform(train_df)
        X_val = self.encoder.transform(val_df)
        X_test = self.encoder.transform(test_df)

        y_train = train_df[TARGET_COL].values
        y_val = val_df[TARGET_COL].values
        y_test = test_df[TARGET_COL].values

        print("\n" + "="*60)
        print("STAGE 5: CLASS IMBALANCE METRICS & PERSISTENCE")
        print("="*60)
        # Calculate imbalance metrics
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos_weight = float(neg_count / max(1, pos_count))
        total_train = len(y_train)

        # Scikit-learn balanced weights: n_samples / (n_classes * np.bincount(y))
        class_weight_0 = float(total_train / (2.0 * max(1, neg_count)))
        class_weight_1 = float(total_train / (2.0 * max(1, pos_count)))

        metadata = {
            "split_method": split_method,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_fraud_count": pos_count,
            "train_fraud_rate_pct": round(float(pos_count / max(1, total_train)) * 100, 3),
            "val_fraud_count": int((y_val == 1).sum()),
            "test_fraud_count": int((y_test == 1).sum()),
            "imbalance_metrics": {
                "scale_pos_weight": round(scale_pos_weight, 3),
                "class_weights": {0: round(class_weight_0, 4), 1: round(class_weight_1, 4)},
                "ratio_neg_to_pos": f"{round(scale_pos_weight, 1)} : 1",
            },
            "feature_names": self.encoder.feature_names_out,
            "clean_report": clean_report,
        }

        # Persist processed datasets
        X_train["is_fraud"] = y_train
        X_val["is_fraud"] = y_val
        X_test["is_fraud"] = y_test

        X_train.to_csv(self.output_dir / "train_processed.csv", index=False)
        X_val.to_csv(self.output_dir / "val_processed.csv", index=False)
        X_test.to_csv(self.output_dir / "test_processed.csv", index=False)

        # Also persist unscaled/un-encoded splits with identifiers for human inspection & risk engine
        train_df.to_csv(self.output_dir / "train_with_features.csv", index=False)
        val_df.to_csv(self.output_dir / "val_with_features.csv", index=False)
        test_df.to_csv(self.output_dir / "test_with_features.csv", index=False)

        # Persist encoder and metadata
        self.encoder.save(str(self.output_dir / "encoder.joblib"))
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[PreprocessingPipeline] Saved all processed splits to {self.output_dir}/")
        print(f"[PreprocessingPipeline] scale_pos_weight: {scale_pos_weight:.2f} (Fraud Rate: {metadata['train_fraud_rate_pct']}%)")

        return X_train, X_val, X_test, metadata

    def _split_data(
        self,
        df: pd.DataFrame,
        method: str = "temporal",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partitions dataset temporally to mirror real-world financial transaction streams."""
        n = len(df)
        if method == "temporal":
            # Strict chronological split: past trains present, present predicts future
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))

            train_df = df_sorted.iloc[:train_end].copy()
            val_df = df_sorted.iloc[train_end:val_end].copy()
            test_df = df_sorted.iloc[val_end:].copy()
        else:
            # Stratified random split fallback
            from sklearn.model_selection import train_test_split
            train_df, rem_df = train_test_split(
                df,
                train_size=self.train_ratio,
                stratify=df[TARGET_COL],
                random_state=42,
            )
            rel_val_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_df, test_df = train_test_split(
                rem_df,
                train_size=rel_val_ratio,
                stratify=rem_df[TARGET_COL],
                random_state=42,
            )

        print(f"[Split] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
