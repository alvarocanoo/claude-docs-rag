"""Async HTTP downloader for Anthropic doc pages, with on-disk cache."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from claude_docs_rag.models import DocSource, RawPage

CACHE_DIR = Path("data/raw")
USER_AGENT = "claude-docs-rag/0.1 (+https://github.com/alvarocanoo/claude-docs-rag)"


def _cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{h}.md"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def _fetch_one(client: httpx.AsyncClient, source: DocSource) -> RawPage:
    cache_path = _cache_path(source.url)
    if cache_path.exists():
        return RawPage(
            url=source.url,
            title=source.title,
            section=source.section,
            content=cache_path.read_text(encoding="utf-8"),
        )

    response = await client.get(source.url, timeout=30.0)
    response.raise_for_status()
    text = response.text

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")

    return RawPage(
        url=source.url,
        title=source.title,
        section=source.section,
        content=text,
    )


async def download_all(
    sources: Iterable[DocSource],
    *,
    concurrency: int = 10,
    client: httpx.AsyncClient | None = None,
) -> list[RawPage]:
    """Download all pages, with bounded concurrency and per-URL retry."""
    sources_list = list(sources)
    semaphore = asyncio.Semaphore(concurrency)
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(headers={"User-Agent": USER_AGENT})

    async def _bounded(src: DocSource) -> RawPage:
        async with semaphore:
            return await _fetch_one(client, src)

    try:
        results = await asyncio.gather(
            *(_bounded(s) for s in sources_list),
            return_exceptions=False,
        )
    finally:
        if owns_client:
            await client.aclose()

    return results
