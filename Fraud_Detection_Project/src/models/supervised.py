"""
Supervised Fraud Classification Engine.
Trains, tunes, benchmarks, and persists multiple supervised models:
Logistic Regression, Decision Tree, Random Forest, and XGBoost.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


class SupervisedFraudModels:
    """Trains and benchmarks supervised classification algorithms on imbalanced fraud data."""

    def __init__(self, scale_pos_weight: float = 65.04, models_dir: str = "models"):
        self.scale_pos_weight = scale_pos_weight
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.optimal_thresholds: Dict[str, float] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.predictions_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.best_model_name: Optional[str] = None

    def initialize_models(self):
        """Instantiates all supervised classification candidates."""
        self.models = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                random_state=42,
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                max_depth=12,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                scale_pos_weight=self.scale_pos_weight,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="aucpr",
                random_state=42,
            ),
        }

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Trains all models on training split, optimizes threshold on validation split,
        and computes comprehensive metrics on test split.
        """
        if not self.models:
            self.initialize_models()

        best_test_prauc = -1.0

        for name, model in self.models.items():
            print(f"\n[SupervisedFraudModels] Training {name}...")
            model.fit(X_train, y_train)

            # Predict probabilities
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]

            # Find optimal threshold on validation set (maximizing F1-Score)
            opt_thresh, best_val_f1 = self._optimize_threshold(y_val, val_probs)
            self.optimal_thresholds[name] = opt_thresh

            # Test predictions with both default (0.5) and optimal threshold
            test_preds_default = (test_probs >= 0.5).astype(int)
            test_preds_opt = (test_probs >= opt_thresh).astype(int)

            # Calculate ROC and PR AUC
            roc_auc = float(roc_auc_score(y_test, test_probs))
            pr_auc = float(average_precision_score(y_test, test_probs))

            # Default threshold metrics
            cm_default = confusion_matrix(y_test, test_preds_default)
            tn_d, fp_d, fn_d, tp_d = cm_default.ravel()

            # Optimal threshold metrics
            cm_opt = confusion_matrix(y_test, test_preds_opt)
            tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()

            self.results[name] = {
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "optimal_threshold": round(opt_thresh, 4),
                "default_threshold_metrics": {
                    "accuracy": round(float(accuracy_score(y_test, test_preds_default)), 4),
                    "precision": round(float(precision_score(y_test, test_preds_default, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_test, test_preds_default)), 4),
                    "f1": round(float(f1_score(y_test, test_preds_default)), 4),
                    "confusion_matrix": {"TN": int(tn_d), "FP": int(fp_d), "FN": int(fn_d), "TP": int(tp_d)},
                },
                "tuned_threshold_metrics": {
                    "accuracy": round(float(accuracy_score(y_test, test_preds_opt)), 4),
                    "precision": round(float(precision_score(y_test, test_preds_opt, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_test, test_preds_opt)), 4),
                    "f1": round(float(f1_score(y_test, test_preds_opt)), 4),
                    "confusion_matrix": {"TN": int(tn_o), "FP": int(fp_o), "FN": int(fn_o), "TP": int(tp_o)},
                },
            }

            # Cache predictions for visualization
            fpr, tpr, _ = roc_curve(y_test, test_probs)
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, test_probs)
            self.predictions_cache[name] = {
                "test_probs": test_probs,
                "test_preds_opt": test_preds_opt,
                "fpr": fpr,
                "tpr": tpr,
                "precision_curve": prec_curve,
                "recall_curve": rec_curve,
            }

            print(f"  --> {name} Results:")
            print(f"      ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
            print(f"      Tuned Thresh ({opt_thresh:.2f}): Recall: {self.results[name]['tuned_threshold_metrics']['recall']:.2%} | Precision: {self.results[name]['tuned_threshold_metrics']['precision']:.2%} | F1: {self.results[name]['tuned_threshold_metrics']['f1']:.4f}")

            if pr_auc > best_test_prauc:
                best_test_prauc = pr_auc
                self.best_model_name = name

        print(f"\n[SupervisedFraudModels] Best Performing Model: {self.best_model_name} (PR-AUC: {best_test_prauc:.4f})")
        self.save_models()
        return self.results

    def _optimize_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
        """Searches thresholds [0.05, 0.95] to find the decision boundary maximizing F1."""
        thresholds = np.linspace(0.05, 0.95, 91)
        best_thresh = 0.50
        best_f1 = -1.0

        for t in thresholds:
            preds = (y_probs >= t).astype(int)
            f = f1_score(y_true, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresh = float(t)

        return best_thresh, best_f1

    def save_models(self):
        """Serializes all models and metadata to models/ directory."""
        for name, model in self.models.items():
            filename = name.lower().replace(" ", "_") + ".joblib"
            path = self.models_dir / filename
            joblib.dump(model, path)

        # Save best model separately as canonical production pointer
        if self.best_model_name and self.best_model_name in self.models:
            best_model_path = self.models_dir / "best_supervised_model.joblib"
            joblib.dump(self.models[self.best_model_name], best_model_path)

        # Save metrics and thresholds
        summary = {
            "best_model": self.best_model_name,
            "optimal_thresholds": self.optimal_thresholds,
            "models": self.results,
        }
        with open(self.models_dir / "supervised_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[SupervisedFraudModels] Saved all models and metrics to {self.models_dir}/")
