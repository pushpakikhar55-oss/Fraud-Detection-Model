# Phase 5 Validation Report

**Executed (UTC):** 2026-09-11T20:46:27.289293+00:00  
**Python:** 3.12.14  
**Regression status:** PASSED

## Regression suite

```text
============================= test session starts =============================
platform win32 -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Asus\OneDrive\Desktop\Fraud_Detection_Project\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Asus\OneDrive\Desktop\Fraud_Detection_Project
plugins: anyio-4.15.1
collecting ... collected 14 items

tests/test_api.py::test_health_check PASSED                              [  7%]
tests/test_api.py::test_readiness_check PASSED                           [ 14%]
tests/test_api.py::test_predict_endpoint PASSED                          [ 21%]
tests/test_api.py::test_transaction_ingestion_and_analytics PASSED       [ 28%]
tests/test_deployment_config.py::test_database_path_reads_environment PASSED [ 35%]
tests/test_features.py::test_feature_engineering_outputs PASSED          [ 42%]
tests/test_generator.py::test_generator_schema_validity PASSED           [ 50%]
tests/test_generator.py::test_fraud_scenarios_presence PASSED            [ 57%]
tests/test_models.py::test_supervised_training_and_metrics PASSED        [ 64%]
tests/test_models.py::test_anomaly_detection_engine PASSED               [ 71%]
tests/test_preprocessing.py::test_cleaner_removes_duplicates_and_invalids PASSED [ 78%]
tests/test_preprocessing.py::test_encoder_leakage_free PASSED            [ 85%]
tests/test_risk_engine.py::test_risk_scorer_bounds_and_bands PASSED      [ 92%]
tests/test_risk_engine.py::test_explainer_local_attributions PASSED      [100%]

======================= 14 passed, 5 warnings in 9.67s ========================
```

## Warm prediction benchmark

- Samples: 20 warmed, non-persistent predictions
- Mean latency: 55.77 ms
- Median latency: 55.02 ms
- P95 latency: 62.03 ms
- Range: 50.72-64.80 ms

## Scope

The benchmark exercises the full feature, model, risk-scoring, and explanation path. It uses `persist=False`, so it does not alter SQLite data or alerts.
