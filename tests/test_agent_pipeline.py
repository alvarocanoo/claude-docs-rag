"""Unit tests for citation extraction in the agent pipeline."""

from __future__ import annotations

from claude_docs_rag.agent.pipeline import _extract_citation_ids


def test_extracts_single_id() -> None:
    assert _extract_citation_ids("As [1] explains, the API streams chunks.", max_id=5) == [1]


def test_extracts_multiple_dedup_preserve_order() -> None:
    text = "See [2][5][2][1] for details, and again [5]."
    assert _extract_citation_ids(text, max_id=5) == [2, 5, 1]


def test_ignores_out_of_range_ids() -> None:
    text = "Bogus reference [99] should be skipped, but [3] is fine."
    assert _extract_citation_ids(text, max_id=5) == [3]


def test_ignores_non_numeric_brackets() -> None:
    text = "Not a citation: [foo] or [3a] or [].  Real: [4]."
    assert _extract_citation_ids(text, max_id=5) == [4]


def test_zero_or_negative_ignored() -> None:
    text = "Edge: [0] not allowed; [-1] not allowed; [1] is."
    assert _extract_citation_ids(text, max_id=5) == [1]


def test_empty_text() -> None:
    assert _extract_citation_ids("", max_id=5) == []
