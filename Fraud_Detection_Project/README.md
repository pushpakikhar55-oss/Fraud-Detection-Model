# AI-Based Financial Fraud Detection System

An enterprise-grade, end-to-end Financial Fraud Detection Platform featuring supervised ML classification, unsupervised anomaly detection, hybrid 0–100 risk scoring, explainable AI (SHAP), FastAPI backend, and an interactive monitoring dashboard.

---

## 📌 Project Architecture & High-Level Flow

```mermaid
graph TD
    A[Raw Transaction Ingestion] --> B[Data Preprocessing & Cleaning]
    B --> C[Behavioral & Velocity Feature Engineering]
    C --> D[Temporal Train / Val / Test Partitioning]
    D --> E[Categorical Encoding & Robust Scaling]
    E --> F[Supervised Classification (Phase 2)]
    E --> G[Anomaly Detection Engine (Phase 2)]
    F --> H[Hybrid Risk Scoring 0-100 (Phase 3)]
    G --> H
    H --> I[Explainable AI - SHAP/LIME (Phase 3)]
    H --> J[FastAPI Backend REST Service (Phase 4)]
    J --> K[Interactive Admin Dashboard (Phase 4)]
```

---

## 🚀 Phase 1: Foundation (Completed)
Phase 1 established data generation, schema validation, preprocessing, causal feature extraction, and full statistical EDA.

---

## 🤖 Phase 2: AI / ML Classification & Anomaly Detection (Completed)
Phase 2 established model training, threshold tuning, and multi-model benchmarking (XGBoost 0.9999 PR-AUC, 100% recall; Isolation Forest 72.2% zero-day recall).

---

## 🧠 Phase 3: Intelligence & Explainable AI (Completed)
Phase 3 built the hybrid 0–100 risk scoring engine, operational decision tiers (Low: Approve, Medium: Review, High: Block), and SHAP-based local/global explainability narratives.

---

## 🌐 Phase 4: Application & Monitoring Dashboard (Completed)

Phase 4 operationalized the AI pipeline into a full-stack platform:
1. **Module 9: Backend REST API & Persistent Database**:
   - **FastAPI** service with asynchronous endpoints for prediction, transaction ingestion, alert triage, and analytics.
   - **SQLite Database** with Write-Ahead Logging (WAL) for persistent transaction storage, alert queues, and audit logging.
   - Sub-50ms real-time inference pipeline loading pre-fitted transformers, XGBoost, and Isolation Forest.
2. **Module 10: Interactive Admin Monitoring Dashboard**:
   - Single-Page Web Dashboard served at `http://127.0.0.1:8000/`.
   - **Executive KPI Cards**: Live counters for total volume, high-risk blocks, review queue, and financial loss prevented.
   - **Interactive Risk Simulator**: Test custom transactions in real-time with animated score gauges, decision pills, and SHAP narrative reasons.
   - **Security Alert Center**: Live triage queue for flagged transactions with one-click actions (`Confirm Fraud`, `Dismiss`).
   - **Transaction Ledger**: Searchable, filterable audit table with one-click deep-dive modal.
   - **Dynamic Visualizations**: Chart.js charts for risk band volume distribution and fraud by transaction channel.

---

## ✅ Phase 5: Validation & Benchmarking (Completed)

Phase 5 adds a reproducible regression-test and benchmark runner. Execute `python run_phase5.py` to run all 13 tests, measure warmed end-to-end prediction latency without changing the SQLite ledger, and create `reports/phase5_validation_report.md`.

---

## 🖥️ Phase 6: Local Application Runtime (Completed)

The platform runs directly on your computer at `http://127.0.0.1:8000/`. Its SQLite database remains local at `data/fraud_platform.db` by default.

---

## 📈 Phase 7: Observability & Operations (Completed)

Every API response now includes an `X-Request-ID` for tracing and `X-Response-Time-Ms` for latency inspection. The API logs each request's method, route, status, and duration. Use `GET /health` for a lightweight liveness probe and `GET /ready` to verify that SQLite plus the model artifacts are loaded and usable.

Run the two added phases directly:

```powershell
.\.venv\Scripts\python.exe run_phase6.py
.\.venv\Scripts\python.exe run_phase7.py
```

---

## 🔐 Phase 8: API Security Hardening (Completed)

Phase 8 rejects invalid scoring inputs before inference, adds browser security headers, and prevents API responses containing transaction data from being cached. Run the verification with:

```powershell
.\.venv\Scripts\python.exe run_phase8.py
```

---

## 🛠️ Quickstart & Launching the Web Platform

### 1. Run Phase 5 Validation (Tests + Benchmark)
```bash
python run_phase5.py
```

### 2. Launch the Web Application
```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Once started, navigate to:
- **Interactive Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger REST API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📁 Repository Structure

```
Fraud_Detection_Project/
├── data/
│   ├── raw/
│   │   └── transactions_raw.csv         # Raw 50,000 transactions
│   └── processed/
│       ├── train_processed.csv          # 35,000 encoded & scaled training samples
│       ├── val_processed.csv            # 7,500 validation samples
│       ├── test_processed.csv           # 7,500 test samples
│       ├── train_with_features.csv      # Unscaled train data with features
│       ├── val_with_features.csv        # Unscaled val data with features
│       ├── test_with_features.csv       # Unscaled test data with features
│       ├── encoder.joblib               # Fitted OneHotEncoder & RobustScaler
│       └── metadata.json                # Partitioning and class weight metadata
├── src/
│   ├── data/
│   │   ├── schema.py                    # Schema definitions and data validation
│   │   ├── generator.py                 # Multi-scenario synthetic transaction generator
│   │   └── loader.py                    # Ingestion and storage manager
│   ├── preprocessing/
│   │   ├── cleaner.py                   # Deduplication and imputation
│   │   ├── encoder.py                   # Categorical & numerical transformers
│   │   └── pipeline.py                  # End-to-end preprocessing pipeline
│   ├── features/
│   │   ├── engineer.py                  # Causal behavioral and velocity engine
│   │   └── registry.py                  # Feature definitions and fraud rationale
│   └── eda/
│       └── analyzer.py                  # Statistical profiling and chart generator
├── reports/
│   ├── figures/                         # High-resolution PNG visual charts
│   ├── data_dictionary.md               # Canonical field and feature dictionary
│   └── eda_report.md                    # Comprehensive EDA summary and findings
├── tests/
│   ├── test_generator.py                # Generator schema & scenario tests
│   ├── test_preprocessing.py            # Preprocessing & leakage prevention tests
│   └── test_features.py                 # Feature engineering integrity tests
├── requirements.txt                     # Project dependencies
├── run_phase1.py                        # Master CLI script executing Phase 1
├── run_phase5.py                        # Regression-test and benchmark runner
├── run_phase6.py                        # Local application runtime verification
├── run_phase7.py                        # Observability and operational probe verification
├── run_phase8.py                        # API security-hardening verification
└── README.md                            # Documentation and architecture guide
```

---

## 🛠️ Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Tests
```bash
python -m pytest tests/
```

### 3. Execute Phase 1 Pipeline
```bash
python run_phase1.py
```
This generates the raw dataset, processes features, partitions into train/val/test splits, computes class weights, and outputs all figures and reports.

---

## ✅ Project Status

All eight development phases are implemented. Run `python run_phase5.py` through `python run_phase8.py` before demonstrations or releases to produce current validation records.
