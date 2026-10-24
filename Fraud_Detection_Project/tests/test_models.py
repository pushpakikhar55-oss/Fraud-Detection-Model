"""
Unit tests for supervised classification and anomaly detection models.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from src.models.supervised import SupervisedFraudModels
from src.models.anomaly import AnomalyDetectionEngine


@pytest.fixture
def dummy_data():
    """Generates synthetic training and test features for fast unit tests."""
    np.random.seed(42)
    n_train, n_val, n_test = 200, 50, 50
    n_features = 10
    
    X_train = pd.DataFrame(np.random.randn(n_train, n_features), columns=[f"f_{i}" for i in range(n_features)])
    y_train = (np.random.rand(n_train) > 0.85).astype(int)

    X_val = pd.DataFrame(np.random.randn(n_val, n_features), columns=[f"f_{i}" for i in range(n_features)])
    y_val = (np.random.rand(n_val) > 0.85).astype(int)

    X_test = pd.DataFrame(np.random.randn(n_test, n_features), columns=[f"f_{i}" for i in range(n_features)])
    y_test = (np.random.rand(n_test) > 0.85).astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def test_supervised_training_and_metrics(dummy_data):
    """Tests supervised models initialization, fit, threshold optimization, and probability bounds."""
    X_train, y_train, X_val, y_val, X_test, y_test = dummy_data
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        sfm = SupervisedFraudModels(scale_pos_weight=5.0, models_dir=tmp_dir)
        results = sfm.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

        assert len(results) == 4
        assert "XGBoost" in results
        assert "Random Forest" in results

        for name, data in results.items():
            assert 0.0 <= data["roc_auc"] <= 1.0
            assert 0.0 <= data["pr_auc"] <= 1.0
            assert 0.05 <= data["optimal_threshold"] <= 0.95

            # Verify predictions cache probabilities
            probs = sfm.predictions_cache[name]["test_probs"]
            assert len(probs) == len(y_test)
            assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_anomaly_detection_engine(dummy_data):
    """Tests Isolation Forest fit, 0-1 score normalization, and evaluation."""
    X_train, _, _, _, X_test, y_test = dummy_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = AnomalyDetectionEngine(contamination=0.10, models_dir=tmp_dir)
        engine.fit(X_train)
        
        scores = engine.compute_anomaly_score(X_test)
        assert len(scores) == len(X_test)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

        metrics = engine.evaluate_test_set(X_test, y_test)
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0
