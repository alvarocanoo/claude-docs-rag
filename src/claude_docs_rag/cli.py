"""CLI entry point."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

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


if __name__ == "__main__":
    app()
