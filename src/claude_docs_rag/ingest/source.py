"""Parse the Anthropic llms.txt index into a list of pages to download.

Format reminder (Anthropic uses the llms.txt convention):
    ## English
    ### <section name>
    - [Page title](https://platform.claude.com/docs/en/.../page.md) - optional description
"""

from __future__ import annotations

import re

import httpx

from claude_docs_rag.models import DocSource

LLMS_INDEX_URL = "https://platform.claude.com/llms.txt"

# Pull bullets of shape:  - [Title](https://platform.claude.com/docs/en/...md) - optional tail
_BULLET_RE = re.compile(
    r"^\s*-\s+\[(?P<title>[^\]]+)\]\((?P<url>https?://[^)]+\.md)\)",
    re.MULTILINE,
)


async def fetch_index(client: httpx.AsyncClient, url: str = LLMS_INDEX_URL) -> str:
    response = await client.get(url, timeout=30.0)
    response.raise_for_status()
    return response.text


def parse_index(text: str, lang_prefix: str = "/docs/en/") -> list[DocSource]:
    """Yield DocSource entries from the index, scoped to a language.

    Scope to English only by default — the multilingual variants are
    pointers ("Visit website for content") and don't have .md endpoints.
    """
    sources: list[DocSource] = []
    current_section = ""
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            current_section = stripped[4:].strip()
            continue
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            continue

        match = _BULLET_RE.match(line)
        if not match:
            continue

        url = match.group("url")
        if lang_prefix not in url or url in seen:
            continue

        seen.add(url)
        sources.append(
            DocSource(
                url=url,
                title=match.group("title").strip(),
                section=current_section,
            )
        )

    return sources
