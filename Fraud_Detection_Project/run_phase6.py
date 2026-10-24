"""Phase 6: validate the local FastAPI runtime without Docker."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "phase6_local_runtime_report.md"


def main() -> int:
    """Verify the local dashboard, liveness endpoint, and readiness endpoint."""
    print("PHASE 6: LOCAL APPLICATION RUNTIME")
    with TestClient(app) as client:
        health = client.get("/health")
        dashboard = client.get("/")
        ready = client.get("/ready")

    checks = {
        "health_endpoint": health.status_code == 200 and health.json().get("status") == "healthy",
        "dashboard": dashboard.status_code == 200,
        "readiness_endpoint": ready.status_code == 200 and ready.json().get("status") == "ready",
    }
    passed = all(checks.values())
    REPORT_PATH.write_text(
        "# Phase 6 Local Runtime Report\n\n"
        f"**Executed (UTC):** {datetime.now(timezone.utc).isoformat()}  \n"
        f"**Status:** {'PASSED' if passed else 'FAILED'}\n\n"
        "## Verification\n\n"
        + "\n".join(f"- {'PASS' if ok else 'FAIL'}: `{name}`" for name, ok in checks.items())
        + "\n\n## Local launch\n\n"
        "```powershell\n.\\.venv\\Scripts\\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000\n```\n"
        "Open `http://127.0.0.1:8000/` after launch.\n",
        encoding="utf-8",
    )
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAILED'}] {name}")
    print(f"Report: {REPORT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
