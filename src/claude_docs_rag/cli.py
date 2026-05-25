"""CLI entry point — placeholder until ingest/query commands land."""

from __future__ import annotations

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
        "[bold yellow]Scaffolding phase[/bold yellow] — ingest pipeline not yet implemented."
    )


if __name__ == "__main__":
    app()
