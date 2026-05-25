"""CLI entry point."""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console

# Windows tweaks (no-ops elsewhere):
# 1. psycopg async is incompatible with the default ProactorEventLoop.
# 2. Console defaults to cp1252, which crashes on UTF-8 output (e.g. → arrows).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(help="claude-docs-rag CLI")
console = Console()


@app.command()
def version() -> None:
    """Print package version."""
    from claude_docs_rag import __version__

    console.print(f"claude-docs-rag v{__version__}")


@app.command()
def status() -> None:
    """Show current scaffolding status."""
    console.print(
        "[bold green]Ingest pipeline ready.[/bold green] Run `cdrag init-db` then `cdrag ingest`."
    )


@app.command("init-db")
def init_db() -> None:
    """Create the pgvector schema (idempotent)."""
    from claude_docs_rag.storage.vector_store import apply_schema, count_documents

    async def _run() -> None:
        await apply_schema()
        n = await count_documents()
        console.print(f"[green]Schema applied.[/green] documents row count: {n}")

    asyncio.run(_run())


@app.command("check-db")
def check_db() -> None:
    """Diagnostic: print extensions, schema columns and row count."""
    from claude_docs_rag.storage.vector_store import describe_storage

    async def _run() -> None:
        info = await describe_storage()
        console.print("[bold]Extensions[/bold]")
        for name, ver in info.extensions.items():
            console.print(f"  - {name} {ver}")
        console.print("[bold]documents columns[/bold]")
        for name, dtype in info.columns:
            console.print(f"  - {name:18} {dtype}")
        console.print(f"[bold]row count[/bold]: {info.documents_count}")

    asyncio.run(_run())


@app.command()
def ingest(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Only ingest the first N pages."),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Parallel downloads."),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Embedding batch size."),
) -> None:
    """Run the full ingest pipeline."""
    from claude_docs_rag.ingest.pipeline import run_ingest

    report = asyncio.run(
        run_ingest(limit=limit, concurrency=concurrency, embed_batch_size=batch_size)
    )
    console.print()
    console.print("[bold]Ingest complete[/bold]")
    console.print(f"  pages_indexed   = {report.pages_indexed}")
    console.print(f"  pages_downloaded= {report.pages_downloaded}")
    console.print(f"  chunks_created  = {report.chunks_created}")
    console.print(f"  chunks_embedded = {report.chunks_embedded}")
    console.print(f"  chunks_stored   = {report.chunks_stored}")
    console.print(f"  elapsed         = {report.elapsed_seconds:.1f}s")


@app.command("build-bm25")
def build_bm25() -> None:
    """Build a BM25 index from the documents table and persist it to disk."""
    from claude_docs_rag.retrieval.sparse import build_index
    from claude_docs_rag.storage.vector_store import iter_corpus

    async def _run() -> list[tuple[str, int, str]]:
        return await iter_corpus()

    corpus = asyncio.run(_run())
    console.print(f"Loaded {len(corpus)} chunks from Postgres. Building BM25 index...")
    build_index(corpus)
    console.print("[green]BM25 index built and persisted under data/bm25_index/[/green]")


@app.command()
def search(query: str, k: int = typer.Option(5, "--k", help="Number of results to print.")) -> None:
    """Hybrid search (dense + BM25 + RRF + cross-encoder rerank)."""
    from claude_docs_rag.retrieval.hybrid import hybrid_search

    results = asyncio.run(hybrid_search(query, top_k_rerank=k))
    console.print(f"\n[bold]Q:[/bold] {query}")
    if not results:
        console.print("[red]No results.[/red]")
        return
    for i, r in enumerate(results, 1):
        console.print(
            f"\n  {i}. [rerank={r.rerank_score:+.3f} fusion={r.fusion_score:.3f}] {r.title}"
        )
        console.print(f"     section: {r.section_path[:90]}")
        console.print(f"     url:     {r.source_url}")
        preview = " ".join(r.content.split())[:200]
        console.print(f"     excerpt: {preview}...")


if __name__ == "__main__":
    app()
