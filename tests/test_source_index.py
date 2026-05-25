"""Unit tests for the llms.txt index parser."""

from __future__ import annotations

from claude_docs_rag.ingest.source import parse_index

SAMPLE = """
## English

### Messages

- [Overview](https://platform.claude.com/docs/en/build/overview.md) - Build overview
- [Quickstart](https://platform.claude.com/docs/en/get-started.md) - Get started
- [German page](https://platform.claude.com/docs/de/some-page.md) - non-English, must be filtered

### Tools

- [Tool use](https://platform.claude.com/docs/en/tools/use.md)
- [Duplicate](https://platform.claude.com/docs/en/tools/use.md) - same URL appears twice
"""


def test_parses_english_only() -> None:
    sources = parse_index(SAMPLE)
    urls = {s.url for s in sources}
    assert "https://platform.claude.com/docs/de/some-page.md" not in urls
    assert "https://platform.claude.com/docs/en/build/overview.md" in urls


def test_deduplicates() -> None:
    sources = parse_index(SAMPLE)
    urls = [s.url for s in sources]
    assert len(urls) == len(set(urls))


def test_captures_section_and_title() -> None:
    sources = parse_index(SAMPLE)
    overview = next(s for s in sources if s.url.endswith("/overview.md"))
    assert overview.title == "Overview"
    assert overview.section == "Messages"

    tool = next(s for s in sources if s.url.endswith("/tools/use.md"))
    assert tool.section == "Tools"
