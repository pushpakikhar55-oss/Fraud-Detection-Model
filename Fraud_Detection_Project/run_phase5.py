"""Phase 5 validation runner for the fraud-detection platform."""
from __future__ import annotations

import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "phase5_validation_report.md"


def run_tests() -> tuple[bool, str]:
    """Run all project tests with the current Python interpreter."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-v", "--disable-warnings"],
        cwd=ROOT, text=True, capture_output=True,
    )
    return result.returncode == 0, (result.stdout + "\n" + result.stderr).strip()


def benchmark_predictions(iterations: int = 20) -> dict[str, float]:
    """Benchmark warmed predictions without writing transactions or alerts."""
    from src.api.service import InferenceService

    service = InferenceService.get_instance()
    payload = {
        "customer_id": "PHASE5_BENCHMARK", "amount": 125.0,
        "transaction_type": "PAYMENT", "merchant_category": "retail",
        "payment_method": "credit_card", "old_balance": 2000.0,
        "is_night": 0, "is_new_device": 0, "is_new_location": 0,
        "tx_count_1h": 1, "customer_hist_mean": 120.0,
    }
    service.process_transaction(payload, persist=False)  # warm model and SHAP caches
    samples_ms = []
    for _ in range(iterations):
        started = time.perf_counter()
        service.process_transaction(payload, persist=False)
        samples_ms.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples_ms)
    return {
        "iterations": iterations, "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[max(0, int(round(0.95 * len(ordered))) - 1)],
        "min_ms": min(samples_ms), "max_ms": max(samples_ms),
    }


def write_report(tests_passed: bool, test_output: str, latency: dict[str, float] | None) -> None:
    """Persist an auditable record of this validation run."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    latency_section = "Not run because regression tests failed."
    if latency:
        latency_section = (
            f"- Samples: {latency['iterations']} warmed, non-persistent predictions\n"
            f"- Mean latency: {latency['mean_ms']:.2f} ms\n"
            f"- Median latency: {latency['median_ms']:.2f} ms\n"
            f"- P95 latency: {latency['p95_ms']:.2f} ms\n"
            f"- Range: {latency['min_ms']:.2f}-{latency['max_ms']:.2f} ms"
        )
    REPORT_PATH.write_text(
        f"# Phase 5 Validation Report\n\n"
        f"**Executed (UTC):** {datetime.now(timezone.utc).isoformat()}  \n"
        f"**Python:** {sys.version.split()[0]}  \n"
        f"**Regression status:** {'PASSED' if tests_passed else 'FAILED'}\n\n"
        f"## Regression suite\n\n```text\n{test_output}\n```\n\n"
        f"## Warm prediction benchmark\n\n{latency_section}\n\n"
        "## Scope\n\nThe benchmark exercises the full feature, model, risk-scoring, and "
        "explanation path. It uses `persist=False`, so it does not alter SQLite data or alerts.\n",
        encoding="utf-8",
    )


def main() -> int:
    print("PHASE 5: REGRESSION TESTS AND WARM INFERENCE BENCHMARK")
    tests_passed, test_output = run_tests()
    print(test_output)
    latency = benchmark_predictions() if tests_passed else None
    write_report(tests_passed, test_output, latency)
    print(f"\nValidation report: {REPORT_PATH}")
    if latency:
        print(f"Mean warm prediction latency: {latency['mean_ms']:.2f} ms")
    return 0 if tests_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
