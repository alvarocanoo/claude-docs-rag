"""Unit tests for the header-aware chunker."""

from __future__ import annotations

from claude_docs_rag.ingest.chunker import chunk_page
from claude_docs_rag.models import RawPage


def _page(content: str) -> RawPage:
    return RawPage(
        url="https://example/docs/test.md",
        title="Test",
        section="Build",
        content=content,
    )


def test_short_page_yields_single_chunk() -> None:
    md = "# Title\n\n## Intro\nA short paragraph about something."
    chunks = chunk_page(_page(md))
    assert len(chunks) == 1
    assert "Intro" in chunks[0].section_path
    assert chunks[0].chunk_index == 0
    assert chunks[0].source_url == "https://example/docs/test.md"


def test_headers_become_breadcrumb() -> None:
    md = (
        "# Page\n\n"
        "## Section A\nFirst section body, with enough text to count as real content.\n\n"
        "### Subsection A1\nMore detail in A1, also reasonably long for testing.\n\n"
        "## Section B\nSecond section body, distinct from A.\n"
    )
    chunks = chunk_page(_page(md), min_tokens=0)  # disable merge: nothing is < 0
    breadcrumbs = [c.section_path for c in chunks]
    assert any("Section A" in b and "Subsection A1" in b for b in breadcrumbs)
    assert any("Section B" in b for b in breadcrumbs)


def test_oversized_section_is_window_split() -> None:
    big_paragraph = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200).strip()
    md = f"# Huge\n\n## Body\n{big_paragraph}"
    chunks = chunk_page(_page(md), target_tokens=200, overlap_tokens=40)
    assert len(chunks) >= 2
    assert all(c.content_tokens <= 260 for c in chunks)


def test_unique_chunk_indices() -> None:
    md = "## A\nfoo bar baz\n\n## B\nqux quux\n\n## C\ncorge grault"
    chunks = chunk_page(_page(md), min_tokens=0)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(indices)))


def test_empty_page_safe() -> None:
    chunks = chunk_page(_page(""))
    assert chunks == []


def test_no_headers_yields_one_chunk() -> None:
    md = "Just plain text. No headers anywhere."
    chunks = chunk_page(_page(md))
    assert len(chunks) == 1
    assert "Just plain text" in chunks[0].content
