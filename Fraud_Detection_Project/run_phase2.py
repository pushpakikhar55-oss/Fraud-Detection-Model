"""
Phase 2 Master Pipeline Execution Script.
Executes Modules 5 and 6:
  - Module 5: Supervised Fraud Classification (LogReg, Decision Tree, Random Forest, XGBoost)
              Threshold Optimization and Evaluation on Imbalanced Metrics
  - Module 6: Unsupervised Anomaly Detection (Isolation Forest & Novelty Risk Scoring)
  - Comparative Evaluation, Chart Generation, Model Persistence
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.supervised import SupervisedFraudModels
from src.models.anomaly import AnomalyDetectionEngine
from src.models.evaluator import ModelEvaluator


def main():
    print("\n" + "#"*70)
    print("AI-BASED FINANCIAL FRAUD DETECTION SYSTEM")
    print("PHASE 2: AI / ML (Modules 5 & 6)")
    print("#"*70)

    # -------------------------------------------------------------
    # 1. LOAD PROCESSED DATASETS
    # -------------------------------------------------------------
    print("\n>>> [STEP 1: LOADING PROCESSED SPLITS]")
    data_dir = Path("data/processed")
    if not (data_dir / "train_processed.csv").exists():
        raise FileNotFoundError("Processed datasets not found in data/processed/. Please execute run_phase1.py first.")

    train_df = pd.read_csv(data_dir / "train_processed.csv")
    val_df = pd.read_csv(data_dir / "val_processed.csv")
    test_df = pd.read_csv(data_dir / "test_processed.csv")

    test_unprocessed = None
    if (data_dir / "test_with_features.csv").exists():
        test_unprocessed = pd.read_csv(data_dir / "test_with_features.csv")

    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    scale_pos_weight = metadata.get("imbalance_metrics", {}).get("scale_pos_weight", 65.04)

    X_train = train_df.drop(columns=["is_fraud"])
    y_train = train_df["is_fraud"].values

    X_val = val_df.drop(columns=["is_fraud"])
    y_val = val_df["is_fraud"].values

    X_test = test_df.drop(columns=["is_fraud"])
    y_test = test_df["is_fraud"].values

    print(f"Loaded feature matrix: {X_train.shape[1]} features")
    print(f"  Train: {X_train.shape[0]:,} samples | Val: {X_val.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")
    print(f"  Class Imbalance Weight (scale_pos_weight): {scale_pos_weight:.2f}")

    # -------------------------------------------------------------
    # 2. MODULE 5: SUPERVISED CLASSIFICATION
    # -------------------------------------------------------------
    print("\n>>> [STEP 2: TRAINING & BENCHMARKING SUPERVISED CLASSIFIERS]")
    supervised_engine = SupervisedFraudModels(scale_pos_weight=scale_pos_weight, models_dir="models")
    supervised_results = supervised_engine.train_and_evaluate(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    # -------------------------------------------------------------
    # 3. MODULE 6: UNSUPERVISED ANOMALY DETECTION
    # -------------------------------------------------------------
    print("\n>>> [STEP 3: TRAINING UNSUPERVISED ANOMALY DETECTOR]")
    anomaly_engine = AnomalyDetectionEngine(contamination=0.02, models_dir="models")
    anomaly_engine.fit(X_train)
    anomaly_results = anomaly_engine.evaluate_test_set(
        X_test=X_test,
        y_test=y_test,
        test_unprocessed=test_unprocessed,
    )
    anomaly_engine.save()

    # -------------------------------------------------------------
    # 4. EVALUATION CHARTS & BENCHMARK REPORT
    # -------------------------------------------------------------
    print("\n>>> [STEP 4: GENERATING COMPARATIVE VISUALIZATIONS & REPORT]")
    evaluator = ModelEvaluator(figures_dir="reports/figures", reports_dir="reports")
    evaluator.plot_all_comparisons(
        supervised_results=supervised_results,
        supervised_cache=supervised_engine.predictions_cache,
        anomaly_engine=anomaly_engine,
        X_test=X_test,
        y_test=y_test,
    )
    evaluator.write_model_comparison_report(
        supervised_results=supervised_results,
        anomaly_results=anomaly_results,
        best_model_name=supervised_engine.best_model_name,
    )

    # -------------------------------------------------------------
    # 5. DELIVERABLES VERIFICATION
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 2 EXECUTION SUMMARY & DELIVERABLES VERIFICATION")
    print("="*70)

    expected_files = [
        "models/logistic_regression.joblib",
        "models/decision_tree.joblib",
        "models/random_forest.joblib",
        "models/xgboost.joblib",
        "models/best_supervised_model.joblib",
        "models/isolation_forest.joblib",
        "models/supervised_metrics.json",
        "models/anomaly_metrics.json",
        "reports/figures/roc_curves_comparison.png",
        "reports/figures/pr_curves_comparison.png",
        "reports/figures/confusion_matrices.png",
        "reports/figures/anomaly_score_distribution.png",
        "reports/figures/supervised_vs_anomaly_scatter.png",
        "reports/model_comparison_report.md",
    ]

    all_exist = True
    for fpath in expected_files:
        exists = Path(fpath).exists()
        status = "[OK]" if exists else "[MISSING]"
        if not exists:
            all_exist = False
        print(f"  {status} {fpath}")

    print(f"\nBest Supervised Model: {supervised_engine.best_model_name}")
    print(f"  PR-AUC: {supervised_results[supervised_engine.best_model_name]['pr_auc']:.4f}")
    print(f"  ROC-AUC: {supervised_results[supervised_engine.best_model_name]['roc_auc']:.4f}")
    print(f"  Optimal Threshold: {supervised_results[supervised_engine.best_model_name]['optimal_threshold']:.2f}")

    if all_exist:
        print("\n>>> PHASE 2 SUCCESSFULLY COMPLETED! ALL DELIVERABLES GENERATED.")
    else:
        print("\n>>> WARNING: Some deliverables were missing.")


if __name__ == "__main__":
    main()
