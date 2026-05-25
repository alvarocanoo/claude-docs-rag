"""Smoke tests — import surface and settings load."""

from __future__ import annotations


def test_package_imports() -> None:
    import claude_docs_rag

    assert claude_docs_rag.__version__ == "0.1.0"


def test_settings_load_with_defaults() -> None:
    from claude_docs_rag.settings import Settings

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.postgres_port == 5432
    assert s.top_k_retrieval == 20
    assert s.top_k_rerank == 5
    assert s.postgres_dsn.startswith("postgresql://")


def test_cli_version_runs() -> None:
    from typer.testing import CliRunner

    from claude_docs_rag.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
