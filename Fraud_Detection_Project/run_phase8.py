"""Phase 8: security-hardening verification for the FastAPI service."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "phase8_security_report.md"


def main() -> int:
    """Verify defensive headers and request validation without writing data."""
    print("PHASE 8: API SECURITY HARDENING")
    with TestClient(app) as client:
        api_response = client.get("/api/analytics")
        invalid_request = client.post("/api/predict", json={"amount": -100, "tx_count_1h": 0})

    expected_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "same-origin",
        "Cache-Control": "no-store",
    }
    checks = {
        "api_response": api_response.status_code == 200,
        "invalid_payload_rejected": invalid_request.status_code == 422,
        **{
            f"header_{name.lower().replace('-', '_')}": api_response.headers.get(name) == value
            for name, value in expected_headers.items()
        },
    }
    passed = all(checks.values())
    REPORT_PATH.write_text(
        "# Phase 8 Security Report\n\n"
        f"**Executed (UTC):** {datetime.now(timezone.utc).isoformat()}  \n"
        f"**Status:** {'PASSED' if passed else 'FAILED'}\n\n"
        "## Verification\n\n"
        + "\n".join(f"- {'PASS' if ok else 'FAIL'}: `{name}`" for name, ok in checks.items())
        + "\n\n## Controls\n\n"
        "- Invalid monetary, flag, and velocity inputs are rejected by the API schema.\n"
        "- API responses are marked `no-store` to avoid caching sensitive transaction data.\n"
        "- Browser responses include anti-sniffing, anti-framing, and referrer controls.\n",
        encoding="utf-8",
    )
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAILED'}] {name}")
    print(f"Report: {REPORT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
