"""Unit tests for the pure scoring helpers in the eval runner."""

from __future__ import annotations

from claude_docs_rag.evals.runner import _citation_match, _percentile, _topic_coverage

# ---------- _topic_coverage ----------


def test_topic_coverage_all_present() -> None:
    score = _topic_coverage(
        "Streaming uses SSE events from messages.stream.", ["streaming", "SSE", "messages.stream"]
    )
    assert score == 1.0


def test_topic_coverage_partial() -> None:
    score = _topic_coverage("Streaming uses SSE.", ["streaming", "SSE", "messages.stream"])
    assert abs(score - 2 / 3) < 1e-9


def test_topic_coverage_case_insensitive() -> None:
    assert _topic_coverage("STREAMING", ["streaming"]) == 1.0


def test_topic_coverage_no_expected_returns_one() -> None:
    assert _topic_coverage("anything", []) == 1.0


# ---------- _citation_match ----------


def test_citation_match_pattern_in_url() -> None:
    urls = ["https://platform.claude.com/docs/en/build-with-claude/streaming.md"]
    assert _citation_match(urls, ["streaming"]) == 1.0


def test_citation_match_no_url_no_pattern_zero() -> None:
    assert _citation_match([], []) == 0.0


def test_citation_match_url_present_no_pattern_required() -> None:
    assert _citation_match(["https://x/y.md"], []) == 1.0


def test_citation_match_pattern_mismatch() -> None:
    urls = ["https://platform.claude.com/docs/en/intro.md"]
    assert _citation_match(urls, ["streaming"]) == 0.0


# ---------- _percentile ----------


def test_percentile_p95() -> None:
    values = [float(i) for i in range(1, 21)]  # 1..20
    assert _percentile(values, 0.95) == 19.0


def test_percentile_p50_median_like() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 0.5) in (3.0, 4.0)  # idx logic floors


def test_percentile_empty() -> None:
    assert _percentile([], 0.95) == 0.0
