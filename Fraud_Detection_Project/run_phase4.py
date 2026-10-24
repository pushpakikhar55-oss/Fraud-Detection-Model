"""
Phase 4 Master Pipeline Execution Script.
Executes Modules 9 and 10:
  - Module 9: Database Setup & Seeding (SQLite with WAL mode)
  - Module 10: Admin Dashboard & Real-Time Alert System Verification
  - API Health Check, Integration Testing, and Launch Guide
"""
import sys
import subprocess
from pathlib import Path
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.database import DatabaseManager


def main():
    print("\n" + "#"*70)
    print("AI-BASED FINANCIAL FRAUD DETECTION SYSTEM")
    print("PHASE 4: APPLICATION & DASHBOARD (Modules 9 & 10)")
    print("#"*70)

    # -------------------------------------------------------------
    # 1. INITIALIZE & SEED DATABASE
    # -------------------------------------------------------------
    print("\n>>> [STEP 1: INITIALIZING SQLITE DATABASE & SEEDING TRANSACTIONS]")
    db = DatabaseManager(db_path="data/fraud_platform.db")

    scored_test_path = Path("data/processed/test_scored_with_risk.csv")
    if scored_test_path.exists():
        df_scored = pd.read_csv(scored_test_path)
        db.seed_from_dataframe(df_scored, max_records=2000)
    else:
        print("[Warning] data/processed/test_scored_with_risk.csv not found. Skipping auto-seed.")

    summary = db.get_analytics_summary()
    print("\nDatabase Status & Analytics:")
    print(f"  - Total Transactions:  {summary['total_transactions']:,}")
    print(f"  - Confirmed Fraud:     {summary['confirmed_fraud_count']:,}")
    print(f"  - High-Risk Blocked:   {summary['high_risk_blocked_count']:,}")
    print(f"  - Active Alerts Queue: {summary['active_alerts_count']:,}")
    print(f"  - Capital Loss Saved:  ${summary['loss_prevented_usd']:,.2f}")

    # -------------------------------------------------------------
    # 2. RUN API AUTOMATED TESTS
    # -------------------------------------------------------------
    print("\n>>> [STEP 2: RUNNING REST API AUTOMATED TESTS]")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_api.py", "-v"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("API integration tests failed.")

    # -------------------------------------------------------------
    # 3. VERIFY APPLICATION ARTIFACTS
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 4 EXECUTION SUMMARY & DELIVERABLES VERIFICATION")
    print("="*70)

    expected_files = [
        "data/fraud_platform.db",
        "src/api/database.py",
        "src/api/models.py",
        "src/api/service.py",
        "src/api/app.py",
        "src/api/static/index.html",
        "tests/test_api.py",
    ]

    all_exist = True
    for fpath in expected_files:
        exists = Path(fpath).exists()
        status = "[OK]" if exists else "[MISSING]"
        if not exists:
            all_exist = False
        print(f"  {status} {fpath}")

    if all_exist:
        print("\n>>> PHASE 4 SUCCESSFULLY COMPLETED! ALL APPLICATION DELIVERABLES READY.")
        print("\nTo launch the live interactive Web Dashboard & REST API, run:")
        print("    python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload")
        print("\nEndpoints:")
        print("  - Interactive Web Dashboard:  http://127.0.0.1:8000/")
        print("  - Interactive Swagger API:    http://127.0.0.1:8000/docs")
        print("  - Analytics Endpoint:         http://127.0.0.1:8000/api/analytics")
        print("  - Security Alerts Endpoint:   http://127.0.0.1:8000/api/alerts")
    else:
        print("\n>>> WARNING: Some deliverables were missing.")


if __name__ == "__main__":
    main()
