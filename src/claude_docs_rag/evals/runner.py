"""Eval runner: loads golden_dataset.jsonl, runs the RAG pipeline on each
question, computes per-row and aggregate metrics, persists to evals/latest.json.

Metrics (per question):
- topic_coverage:    fraction of expected_topics that appear (case-insensitive)
                     in the generated answer text.
- citation_match:    1.0 if any cited URL contains any expected_url_pattern,
                     else 0.0. Strict signal for "did it ground correctly".
- has_citation:      1.0 if at least one bracket citation was emitted.

Aggregate metrics:
- avg_topic_coverage, avg_citation_match, citation_rate,
  avg_latency_seconds, avg_cost_usd, p95_latency_seconds, total_cost_usd.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from claude_docs_rag.agent.pipeline import answer_question

GOLDEN_PATH = Path("evals/golden_dataset.jsonl")
LATEST_PATH = Path("evals/latest.json")


@dataclass
class GoldenItem:
    id: str
    category: str
    question: str
    expected_topics: list[str]
    expected_url_patterns: list[str]


@dataclass
class EvalRow:
    id: str
    category: str
    question: str
    answer: str
    topic_coverage: float
    citation_match: float
    has_citation: float
    latency_seconds: float
    cost_usd: float
    cited_urls: list[str]


@dataclass
class EvalReport:
    n: int
    by_id: list[EvalRow]
    avg_topic_coverage: float
    avg_citation_match: float
    citation_rate: float
    avg_latency_seconds: float
    p95_latency_seconds: float
    avg_cost_usd: float
    total_cost_usd: float
    elapsed_seconds: float
    timestamp: float = field(default_factory=time.time)


def load_golden() -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            GoldenItem(
                id=raw["id"],
                category=raw.get("category", "uncategorised"),
                question=raw["question"],
                expected_topics=list(raw.get("expected_topics", [])),
                expected_url_patterns=list(raw.get("expected_url_patterns", [])),
            )
        )
    return items


def _topic_coverage(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    answer_low = answer.lower()
    matched = sum(1 for t in expected if t.lower() in answer_low)
    return matched / len(expected)


def _citation_match(cited_urls: list[str], expected_patterns: list[str]) -> float:
    if not expected_patterns:
        return 1.0 if cited_urls else 0.0
    return float(any(any(p.lower() in u.lower() for p in expected_patterns) for u in cited_urls))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, min(len(sorted_v) - 1, int(pct * (len(sorted_v) - 1))))
    return sorted_v[idx]


async def run_evals(
    *,
    limit: int | None = None,
    model: str | None = None,
    top_k_rerank: int | None = None,
) -> EvalReport:
    started = time.perf_counter()
    items = load_golden()
    if limit is not None:
        items = items[:limit]

    rows: list[EvalRow] = []
    for it in items:
        result = await answer_question(it.question, model=model, top_k_rerank=top_k_rerank)
        cited_urls = [c.source_url for c in result.citations]
        rows.append(
            EvalRow(
                id=it.id,
                category=it.category,
                question=it.question,
                answer=result.answer,
                topic_coverage=_topic_coverage(result.answer, it.expected_topics),
                citation_match=_citation_match(cited_urls, it.expected_url_patterns),
                has_citation=1.0 if result.citations else 0.0,
                latency_seconds=result.latency_seconds,
                cost_usd=result.call.cost_usd,
                cited_urls=cited_urls,
            )
        )

    latencies = [r.latency_seconds for r in rows]
    costs = [r.cost_usd for r in rows]
    n = max(len(rows), 1)
    report = EvalReport(
        n=len(rows),
        by_id=rows,
        avg_topic_coverage=sum(r.topic_coverage for r in rows) / n,
        avg_citation_match=sum(r.citation_match for r in rows) / n,
        citation_rate=sum(r.has_citation for r in rows) / n,
        avg_latency_seconds=statistics.fmean(latencies) if latencies else 0.0,
        p95_latency_seconds=_percentile(latencies, 0.95),
        avg_cost_usd=statistics.fmean(costs) if costs else 0.0,
        total_cost_usd=sum(costs),
        elapsed_seconds=time.perf_counter() - started,
    )

    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
