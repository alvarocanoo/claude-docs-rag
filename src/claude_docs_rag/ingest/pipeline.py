"""End-to-end ingest orchestrator: index → download → chunk → embed → store."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from claude_docs_rag.ingest.chunker import chunk_page
from claude_docs_rag.ingest.embedder import embed_chunks
from claude_docs_rag.ingest.scraper import download_all
from claude_docs_rag.ingest.source import LLMS_INDEX_URL, fetch_index, parse_index
from claude_docs_rag.models import Chunk, DocSource, RawPage
from claude_docs_rag.storage.vector_store import upsert_chunks

console = Console()


@dataclass
class IngestReport:
    pages_indexed: int
    pages_downloaded: int
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    elapsed_seconds: float


async def run_ingest(
    *,
    limit: int | None = None,
    concurrency: int = 10,
    embed_batch_size: int = 32,
    index_url: str = LLMS_INDEX_URL,
) -> IngestReport:
    started = time.perf_counter()

    async with httpx.AsyncClient() as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            t_idx = progress.add_task("Fetching index", total=1)
            index_text = await fetch_index(client, index_url)
            sources = parse_index(index_text)
            if limit is not None:
                sources = sources[:limit]
            progress.update(t_idx, completed=1, total=1)

            console.print(f"  → {len(sources)} pages to ingest")

            t_dl = progress.add_task("Downloading pages", total=len(sources))
            pages: list[RawPage] = []
            for batch in _chunked(sources, concurrency * 4):
                batch_pages = await download_all(batch, concurrency=concurrency, client=client)
                pages.extend(batch_pages)
                progress.update(t_dl, completed=len(pages))

    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page))

    console.print(f"  → {len(chunks)} chunks produced")

    if chunks:
        embed_chunks(chunks, batch_size=embed_batch_size)

    embedded = sum(1 for c in chunks if c.embedding is not None)
    stored = await upsert_chunks(chunks)

    elapsed = time.perf_counter() - started
    return IngestReport(
        pages_indexed=len(sources),
        pages_downloaded=len(pages),
        chunks_created=len(chunks),
        chunks_embedded=embedded,
        chunks_stored=stored,
        elapsed_seconds=elapsed,
    )


def _chunked(items: list[DocSource], size: int) -> Iterator[list[DocSource]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
