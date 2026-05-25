"""Placeholder regression gate — compares latest eval metrics to baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

THRESHOLD = 0.02  # 2 % regression tolerance

BASELINE = Path("evals/baseline.json")
LATEST = Path("evals/latest.json")


def main() -> int:
    if not BASELINE.exists() or not LATEST.exists():
        print("baseline.json or latest.json missing — skipping (first run).")
        return 0

    base = json.loads(BASELINE.read_text())
    latest = json.loads(LATEST.read_text())

    failed = False
    for metric, base_val in base.items():
        if metric not in latest:
            continue
        new_val = latest[metric]
        if new_val < base_val * (1 - THRESHOLD):
            print(f"REGRESSION: {metric} {base_val:.3f} -> {new_val:.3f}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
