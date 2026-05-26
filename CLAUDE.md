# CLAUDE.md — Instructions for Claude working in this repo

> Project-specific instructions. The global `~/.claude/CLAUDE.md` rules apply on top of these.

## Project

**claude-docs-rag** — production-grade RAG agent over Anthropic Claude API docs. See `README.md` for the why and `docs/DECISIONS.md` for technical rationale.

## Stack (do not silently change — every choice has an ADR in `docs/DECISIONS.md`)

- Python 3.12 (see `.python-version`)
- `uv` for package management — NEVER use `pip install` directly; use `uv add` / `uv sync`.
- FastAPI + uvicorn. SSE streaming on `/ask` is roadmap, not shipped yet.
- Anthropic SDK (`anthropic>=0.104`).
- Postgres + `pgvector` on **Neon serverless** (ADR-007). No local `docker-compose.yml`; DSN comes from `.env`.
- Embeddings: `BAAI/bge-small-en-v1.5` (384-dim, ADR-008). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 MB, ADR-009). Both local CPU.
- BM25: `bm25s`, index persisted under `data/bm25_index/`.
- Frontend: Next.js 16 + React 19 + Tailwind 4 under `web/`. Hits FastAPI cross-origin (CORS in `api/server.py`).
- Observability (roadmap): Langfuse traces — not wired yet.
- Semantic cache (roadmap): Redis — not wired yet.
- Lint: `ruff`. Type-check: `mypy --strict`. Tests: `pytest`. CI in `.github/workflows/ci.yml`.

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

# DB schema (Neon — DSN in .env)
uv run cdrag init-db
uv run cdrag check-db

# Ingest corpus + build BM25 index (idempotent, safe to re-run)
uv run cdrag ingest --concurrency 8 --pages-per-batch 30
uv run cdrag build-bm25

# Run API + frontend locally
uv run cdrag serve --host 127.0.0.1 --port 8000
cd web ; npm install ; npm run dev
```

## Conventions

- Source code lives in `src/claude_docs_rag/`. Tests in `tests/`. Eval datasets in `evals/`.
- Module-level docstrings: one line, only when the module name is not self-explanatory.
- Type hints are mandatory (`mypy --strict` is the CI gate).
- Async by default for I/O (HTTP, DB, Anthropic API).
- Never hardcode secrets. Use `pydantic-settings` reading from `.env`.
- Every Anthropic call goes through `src/claude_docs_rag/agent/client.py` (`create_message()`), which adds tenacity retries and cost accounting via the `PRICING` table. Langfuse traces will plug into the same wrapper when added.

## Architectural decisions — do not violate without an ADR

These are locked decisions. If a task seems to require violating one, STOP and write a new ADR in `docs/DECISIONS.md` first.

1. **Hybrid retrieval is mandatory** — sparse (BM25) + dense (bge-small-en-v1.5) fused with RRF. No pure-semantic shortcut.
2. **Reranking is mandatory** — retriever returns top-20, reranker (`ms-marco-MiniLM-L-6-v2`) cuts to top-5. Don't skip.
3. **Cost is a first-class metric** — every Anthropic call logs input/output/cache token counts and computed USD via `agent/client.py::CallResult`.
4. **Evals are a CI gate** — the job lives in `.github/workflows/ci.yml`; activate the regression gate as soon as `ANTHROPIC_API_KEY` is wired as a repo secret. Merge blocks if faithfulness drops > 2 % vs `evals/baseline.json`.
5. **Langfuse traces** — when added, they plug into `agent/client.py::create_message()`. No untraced LLM call should ship to prod.

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
