"""Regression gate: compare scalar metrics in evals/latest.json against
evals/baseline.json. Fail with exit 1 if any guarded metric drops by more
than TOLERANCE relative to the baseline.

To stay apples-to-apples when CI runs `cdrag eval --limit N` against a
baseline that was produced with all 32 golden Q&A, the baseline metrics
are *re-aggregated over the same N rows that latest covered* (matched by
question id). The README still cites the full 32-question baseline numbers
in its Success metrics table.

Run after `cdrag eval` writes latest.json. See ADR-004 + ADR-010.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

GUARDED_METRICS: tuple[str, ...] = (
    "avg_topic_coverage",
    "avg_citation_match",
    "citation_rate",
)
TOLERANCE = 0.02  # 2 % relative drop allowed

BASELINE = Path("evals/baseline.json")
LATEST = Path("evals/latest.json")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    # utf-8-sig transparently strips the BOM if some tooling wrote one.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the three guarded metrics over a set of EvalRow dicts."""
    if not rows:
        return dict.fromkeys(GUARDED_METRICS, 0.0)
    n = len(rows)
    return {
        "avg_topic_coverage": sum(r["topic_coverage"] for r in rows) / n,
        "avg_citation_match": sum(r["citation_match"] for r in rows) / n,
        "citation_rate": sum(r["has_citation"] for r in rows) / n,
    }


def main() -> int:
    base = _load(BASELINE)
    latest = _load(LATEST)

    base_rows: list[dict[str, Any]] = base.get("by_id", [])
    latest_rows: list[dict[str, Any]] = latest.get("by_id", [])
    latest_ids = {r["id"] for r in latest_rows}
    base_subset = [r for r in base_rows if r["id"] in latest_ids]

    print(
        f"baseline: provider={base.get('provider')} model={base.get('model')} "
        f"n={base.get('n')} (matched subset: {len(base_subset)})"
    )
    print(
        f"latest:   provider={latest.get('provider')} model={latest.get('model')} "
        f"n={latest.get('n')}"
    )
    print(f"tolerance: {TOLERANCE * 100:.1f}% relative drop")
    if not base_subset:
        print("WARNING: no overlapping question ids — baseline subset is empty.")
        return 0

    base_agg = _aggregate(base_subset)
    latest_agg = _aggregate(latest_rows)
    print(
        f"\nLatency (latest): avg={statistics.fmean(r['latency_seconds'] for r in latest_rows):.2f}s\n"
    )

    regressions: list[str] = []
    for metric in GUARDED_METRICS:
        base_val = base_agg[metric]
        new_val = latest_agg[metric]
        floor = base_val * (1 - TOLERANCE)
        if new_val < floor:
            drop_pct = (base_val - new_val) / max(base_val, 1e-9) * 100
            print(
                f"  FAIL  {metric}: {base_val:.3f} -> {new_val:.3f} "
                f"(floor {floor:.3f}, drop {drop_pct:.1f}%)"
            )
            regressions.append(metric)
        else:
            delta = new_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f"  OK    {metric}: {base_val:.3f} -> {new_val:.3f} ({sign}{delta:.3f})")

    if regressions:
        print(f"\nREGRESSED: {', '.join(regressions)}")
        return 1
    print("\nAll guarded metrics within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
