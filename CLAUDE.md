# CLAUDE.md — Instructions for Claude working in this repo

> Project-specific instructions. The global `~/.claude/CLAUDE.md` rules apply on top of these.

## Project

**claude-docs-rag** — production-grade RAG agent over Anthropic Claude API docs. See `README.md` for the why and `docs/DECISIONS.md` for technical rationale.

## Stack (do not silently change)

- Python 3.12 (see `.python-version`)
- `uv` for package management — NEVER use `pip install` directly; use `uv add` / `uv sync`.
- FastAPI + uvicorn (SSE streaming).
- Anthropic SDK (`anthropic>=0.104`).
- Postgres 17 + pgvector (Docker — see `docker-compose.yml`).
- Redis 8 (Docker).
- Embeddings: `BAAI/bge-m3`. Reranker: `BAAI/bge-reranker-v2-m3`. Both local.
- BM25: `bm25s`.
- Observability: Langfuse self-hosted.
- Lint: `ruff`. Type-check: `mypy --strict`. Tests: `pytest`.

## Commands you'll need

```powershell
# Install / sync deps
uv sync

# Add a dep
uv add <pkg>

# Run a script
uv run python -m claude_docs_rag.<module>

# Tests
uv run pytest

# Lint + format
uv run ruff check . ; uv run ruff format .

# Type check
uv run mypy src

# Infra up
docker compose up -d
docker compose down
```

## Conventions

- Source code lives in `src/claude_docs_rag/`. Tests in `tests/`. Eval datasets in `evals/`.
- Module-level docstrings: one line, only when the module name is not self-explanatory.
- Type hints are mandatory (`mypy --strict` is the CI gate).
- Async by default for I/O (HTTP, DB, Anthropic API).
- Never hardcode secrets. Use `pydantic-settings` reading from `.env`.
- Every Anthropic call MUST go through the wrapper in `src/claude_docs_rag/agent/client.py` (when it exists) — it adds Langfuse tracing, retries, and cost accounting.

## Architectural decisions — do not violate without an ADR

These are locked decisions. If a task seems to require violating one, STOP and write a new ADR in `docs/DECISIONS.md` first.

1. **Hybrid retrieval is mandatory** — sparse (BM25) + dense (bge-m3) fused with RRF. No pure-semantic shortcut.
2. **Reranking is mandatory** — retriever returns top-20, reranker cuts to top-5. Don't skip.
3. **Every retrieval/generation step is traced** via Langfuse. No untraced LLM calls.
4. **Evals run in CI on every PR** — merge blocks if faithfulness drops > 2 % vs main.
5. **Cost is a first-class metric** — every response logs token counts and $ estimate.

## What NOT to do

- Don't add new dependencies without justifying in commit message (size, alternatives considered).
- Don't bypass `uv` (no `pip install`, no `requirements.txt`).
- Don't disable type checks or eval gates to make CI green. Fix the root cause.
- Don't introduce abstractions until at least 3 concrete call sites need them.
- Don't write `print()` for runtime logging — use `structlog` or `rich.logging` (when added).

## Verification before declaring "done"

For any non-trivial change:
1. `uv run ruff check .` — must be clean.
2. `uv run mypy src` — must be clean.
3. `uv run pytest` — all green.
4. If touching retrieval or generation: run the eval suite, report the delta.
