"""Phase 7: validate API observability and operational probes."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "phase7_observability_report.md"


def main() -> int:
    """Verify correlation IDs, response timings, liveness, and readiness."""
    print("PHASE 7: OBSERVABILITY AND OPERATIONS")
    supplied_id = "phase7-operational-check"
    with TestClient(app) as client:
        health = client.get("/health", headers={"X-Request-ID": supplied_id})
        ready = client.get("/ready")

    checks = {
        "health_endpoint": health.status_code == 200,
        "readiness_endpoint": ready.status_code == 200 and ready.json().get("model_loaded") is True,
        "request_id_propagation": health.headers.get("X-Request-ID") == supplied_id,
        "response_timing_header": float(health.headers.get("X-Response-Time-Ms", "-1")) >= 0.0,
    }
    passed = all(checks.values())
    REPORT_PATH.write_text(
        "# Phase 7 Observability Report\n\n"
        f"**Executed (UTC):** {datetime.now(timezone.utc).isoformat()}  \n"
        f"**Status:** {'PASSED' if passed else 'FAILED'}\n\n"
        "## Operational checks\n\n"
        + "\n".join(f"- {'PASS' if ok else 'FAIL'}: `{name}`" for name, ok in checks.items())
        + "\n\n## Available probes\n\n"
        "- `GET /health` - lightweight liveness check\n"
        "- `GET /ready` - model artifact and SQLite readiness check\n"
        "- Every response includes `X-Request-ID` and `X-Response-Time-Ms`.\n",
        encoding="utf-8",
    )
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAILED'}] {name}")
    print(f"Report: {REPORT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
