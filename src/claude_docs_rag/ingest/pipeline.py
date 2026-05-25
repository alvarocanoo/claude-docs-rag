"""End-to-end ingest orchestrator: index → download → chunk → embed → store.

Streamed in page-batches so progress is visible, memory stays bounded and a
mid-run failure does not waste the previous batches.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from claude_docs_rag.ingest.chunker import chunk_page
from claude_docs_rag.ingest.embedder import embed_chunks
from claude_docs_rag.ingest.scraper import download_all
from claude_docs_rag.ingest.source import LLMS_INDEX_URL, fetch_index, parse_index
from claude_docs_rag.models import Chunk, DocSource
from claude_docs_rag.storage.vector_store import existing_source_urls, upsert_chunks

console = Console()


@dataclass
class IngestReport:
    pages_indexed: int
    pages_skipped_existing: int
    pages_downloaded: int
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    elapsed_seconds: float


def _chunked(items: list[DocSource], size: int) -> Iterator[list[DocSource]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def run_ingest(
    *,
    limit: int | None = None,
    concurrency: int = 10,
    embed_batch_size: int = 32,
    pages_per_batch: int = 50,
    skip_existing: bool = True,
    index_url: str = LLMS_INDEX_URL,
) -> IngestReport:
    started = time.perf_counter()

    async with httpx.AsyncClient() as client:
        index_text = await fetch_index(client, index_url)
        all_sources = parse_index(index_text)
        if limit is not None:
            all_sources = all_sources[:limit]

        if skip_existing and all_sources:
            already = await existing_source_urls([s.url for s in all_sources])
            sources = [s for s in all_sources if s.url not in already]
            skipped = len(all_sources) - len(sources)
        else:
            sources = all_sources
            skipped = 0

        console.print(
            f"  index: {len(all_sources)} pages | "
            f"to-ingest: {len(sources)} | skipping existing: {skipped}"
        )

        totals = {"pages": 0, "chunks": 0, "embedded": 0, "stored": 0}

        if not sources:
            return IngestReport(
                pages_indexed=len(all_sources),
                pages_skipped_existing=skipped,
                pages_downloaded=0,
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                elapsed_seconds=time.perf_counter() - started,
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            t_pages = progress.add_task("Pages", total=len(sources))
            t_chunks = progress.add_task("Chunks stored", total=None)

            for batch in _chunked(sources, pages_per_batch):
                pages = await download_all(batch, concurrency=concurrency, client=client)
                totals["pages"] += len(pages)

                chunks: list[Chunk] = []
                for page in pages:
                    chunks.extend(chunk_page(page))
                totals["chunks"] += len(chunks)

                if chunks:
                    embed_chunks(chunks, batch_size=embed_batch_size)
                    totals["embedded"] += sum(1 for c in chunks if c.embedding is not None)
                    stored = await upsert_chunks(chunks)
                    totals["stored"] += stored

                progress.update(t_pages, completed=totals["pages"])
                progress.update(t_chunks, completed=totals["stored"])

    return IngestReport(
        pages_indexed=len(all_sources),
        pages_skipped_existing=skipped,
        pages_downloaded=totals["pages"],
        chunks_created=totals["chunks"],
        chunks_embedded=totals["embedded"],
        chunks_stored=totals["stored"],
        elapsed_seconds=time.perf_counter() - started,
    )
