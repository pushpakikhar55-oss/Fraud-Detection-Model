"""
Phase 1 Master Pipeline Execution Script.
Executes Modules 1 through 4:
  - Module 1: Data Collection & High-Fidelity Synthetic Simulation
  - Module 2: Data Preprocessing & Leakage-Free Temporal Partitioning
  - Module 3: Behavioral, Velocity, and Novelty Feature Engineering
  - Module 4: Statistical EDA, Visualizations, and Quality Reporting
"""
import sys
import os
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.eda.analyzer import EDAAnalyzer


def main():
    print("\n" + "#"*70)
    print("AI-BASED FINANCIAL FRAUD DETECTION SYSTEM")
    print("PHASE 1: FOUNDATION (Modules 1 - 4)")
    print("#"*70)

    # -------------------------------------------------------------
    # MODULE 1: DATA COLLECTION & GENERATION
    # -------------------------------------------------------------
    print("\n>>> [MODULE 1: DATA COLLECTION & INGESTION]")
    loader = DataLoader(raw_data_dir="data/raw")
    df_raw = loader.load_or_generate_raw(
        file_name="transactions_raw.csv",
        num_records=50000,
        num_customers=1000,
        fraud_rate=0.015,
        force_regenerate=False,
    )

    # -------------------------------------------------------------
    # MODULES 2 & 3: PREPROCESSING & FEATURE ENGINEERING
    # -------------------------------------------------------------
    print("\n>>> [MODULES 2 & 3: PREPROCESSING & FEATURE ENGINEERING]")
    pipeline = PreprocessingPipeline(
        output_dir="data/processed",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    X_train, X_val, X_test, metadata = pipeline.process(df_raw, split_method="temporal")

    # -------------------------------------------------------------
    # MODULE 4: EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n>>> [MODULE 4: EXPLORATORY DATA ANALYSIS]")
    # Load unscaled dataset with engineered features for interpretable EDA
    import pandas as pd
    df_featured = pd.read_csv("data/processed/train_with_features.csv")
    df_featured["timestamp"] = pd.to_datetime(df_featured["timestamp"])
    
    analyzer = EDAAnalyzer(figures_dir="reports/figures", reports_dir="reports")
    stats = analyzer.run_full_eda(df_featured)

    # -------------------------------------------------------------
    # VERIFY DELIVERABLES
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 1 EXECUTION SUMMARY & DELIVERABLES VERIFICATION")
    print("="*70)
    
    expected_files = [
        "data/raw/transactions_raw.csv",
        "data/processed/train_processed.csv",
        "data/processed/val_processed.csv",
        "data/processed/test_processed.csv",
        "data/processed/encoder.joblib",
        "data/processed/metadata.json",
        "reports/figures/class_imbalance.png",
        "reports/figures/amount_distribution.png",
        "reports/figures/fraud_by_type_and_channel.png",
        "reports/figures/temporal_fraud_patterns.png",
        "reports/figures/feature_correlation_heatmap.png",
        "reports/figures/velocity_and_outliers.png",
        "reports/data_dictionary.md",
        "reports/eda_report.md",
    ]

    all_exist = True
    for fpath in expected_files:
        exists = Path(fpath).exists()
        status = "[OK]" if exists else "[MISSING]"
        if not exists:
            all_exist = False
        print(f"  {status} {fpath}")

    print("\nDataset Summary:")
    print(f"  - Training Set:   {len(X_train):,} samples ({metadata['train_fraud_count']} fraud, {metadata['train_fraud_rate_pct']}%)")
    print(f"  - Validation Set: {len(X_val):,} samples ({metadata['val_fraud_count']} fraud)")
    print(f"  - Testing Set:    {len(X_test):,} samples ({metadata['test_fraud_count']} fraud)")
    print(f"  - Total Features: {len(metadata['feature_names'])}")
    print(f"  - Imbalance Weight (scale_pos_weight): {metadata['imbalance_metrics']['scale_pos_weight']}")

    if all_exist:
        print("\n>>> PHASE 1 SUCCESSFULLY COMPLETED! ALL DELIVERABLES GENERATED.")
    else:
        print("\n>>> WARNING: Some deliverables were missing.")


if __name__ == "__main__":
    main()
