# claude-docs-rag

> Production-grade RAG agent over the Anthropic Claude API documentation. Hybrid retrieval (BM25 + embeddings + reranking), eval suite with CI regression gates, semantic caching and multi-model routing.

### 🔴 Live demo

| Frontend (Next.js on Vercel)                                            | Backend (FastAPI on HF Spaces)                                                                                          |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **<https://claude-docs-rag.vercel.app>** · search page                  | **<https://alvarocano-claude-docs-rag.hf.space>** ([Space page](https://huggingface.co/spaces/alvarocano/claude-docs-rag)) |
| **<https://claude-docs-rag.vercel.app/chat>** · streaming chat UI       | `GET /healthz` · `POST /search` · `POST /ask` · `POST /ask/stream` (SSE)                                                |

**Try the live API directly** (no API key needed):

```bash
# Hybrid retrieval only — ~3-4 s, returns 5 reranked chunks
curl -X POST https://alvarocano-claude-docs-rag.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{"query":"How does prompt caching work?","k":5}'

# End-to-end RAG: retrieval -> Groq Llama 3.1 8B -> answer with [n] citations
# ~6-7 s, costs <$0.0002 per call on Groq paid-tier pricing (free-tier in prod)
curl -X POST https://alvarocano-claude-docs-rag.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How does prompt caching work?","k":5,"max_tokens":400}'
```

**Status**: deployed end-to-end with a real eval baseline committed. Hybrid retrieval over 42,248 chunks of Anthropic Claude docs on Neon Postgres + pgvector. Agent has a pluggable provider ([ADR-010](docs/DECISIONS.md)) — defaults to Groq's free tier (Llama 3.1 8B / 3.3 70B) so the whole stack runs at $0; flip `LLM_PROVIDER=anthropic` for Haiku/Sonnet/Opus. Eval suite baseline ([`evals/baseline.json`](evals/baseline.json)) committed for CI regression gate. CI green (ruff + mypy --strict + pytest 53/53 + hadolint).

![claude-docs-rag chat UI](docs/images/ui-chat.png)

> Real screenshot of `/chat`: SSE token streaming through `POST /ask/stream`. Groq Llama 3.1 8B on free tier, 3.7 s end-to-end, $0.00018 / query, with the agent meta line (`provider/model · tokens · cost · latency`) rendered inline so every answer is auditable.

![claude-docs-rag search UI](docs/images/ui-search.png)

> Real screenshot of `/`: Next.js 16 + Tailwind 4 frontend hitting the FastAPI `/search` endpoint with the BM25 + dense + cross-encoder pipeline against 42,248 chunks of the Anthropic Claude API docs. Latency, rerank scores, and result URLs are unmocked.

---

## Why this exists

Most public RAG demos are toys: single retriever, no evals, no observability, no caching, no cost discipline. This repo is the opposite — a small but **defensible** system where every architectural choice has a documented trade-off and a measurable result.

If you're an engineer reviewing this for hiring purposes, the most interesting files are likely:

- [`docs/DECISIONS.md`](docs/DECISIONS.md) — 11 Architecture Decision Records with trade-offs.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System diagram and request flow.
- [`src/claude_docs_rag/retrieval/hybrid.py`](src/claude_docs_rag/retrieval/hybrid.py) — dense + sparse + RRF + reranker pipeline.
- [`src/claude_docs_rag/agent/pipeline.py`](src/claude_docs_rag/agent/pipeline.py) — citation extraction + prompt caching + cost accounting.
- [`src/claude_docs_rag/evals/runner.py`](src/claude_docs_rag/evals/runner.py) — eval harness for CI regression gate.
- [`evals/golden_dataset.jsonl`](evals/golden_dataset.jsonl) — 32 Q&A across 14 doc categories.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — ruff + mypy --strict + pytest, with eval gate on PRs.

---

## Verified end-to-end

| Component                       | Status | Evidence                                                                                                                                                                  |
|---------------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CI on GitHub Actions            | green  | ruff, mypy --strict (26 files), pytest 40/40, hadolint Dockerfile lint                                                                                                    |
| Neon serverless Postgres        | live   | pgvector 0.8.0 + pg_trgm 1.6, vector(384) HNSW index, region `eu-central-1`                                                                                              |
| Ingest (incremental, idempotent)| live   | 42,248 chunks of `platform.claude.com/docs` in Postgres                                                                                                                  |
| BM25 index (`bm25s`)            | live   | bootstrapped automatically from Postgres on Space cold start (verified locally: 30.9 s)                                                                                  |
| Hybrid retrieval                | **live** | [`/search`](https://alvarocano-claude-docs-rag.hf.space/healthz) via HF Space returns top-K reranked hits in ~3-4 s end-to-end (measured: 5.0 s wall, 4.2 s server)     |
| FastAPI HTTP server             | **live** | `https://alvarocano-claude-docs-rag.hf.space` — `/healthz`, `/search`, `/ask`, `/metrics`, lifespan warmup + BM25 bootstrap                                              |
| Next.js 16 frontend             | **live** | `https://claude-docs-rag.vercel.app` — hits `/search` cross-origin (CORS pinned to Vercel domain)                                                                        |
| Dockerfile + .dockerignore      | live   | multi-stage, non-root, healthcheck; embedder + reranker pre-cached at build → cold start ~5 s instead of 30-60 s; hadolint clean in CI                                  |
| GitHub → HF Space sync          | live   | `.github/workflows/sync-to-hf-space.yml` mirrors `main` to the Space on every push (`HF_TOKEN` secret + `HF_USER` / `HF_SPACE_NAME` vars)                                |
| Agent + citation extraction     | live   | pluggable LLM provider (ADR-010): `LLM_PROVIDER=anthropic` or `groq`. Default in prod is Groq free-tier (`llama-3.1-8b-instant` for high-volume, `llama-3.3-70b-versatile` for spot demos). |
| Eval suite + baseline           | live   | [`evals/baseline.json`](evals/baseline.json): 32 Q&A, Groq 8b-instant, real numbers — see *Success metrics* below                                                         |

---

## Success metrics

Real numbers from [`evals/baseline.json`](evals/baseline.json) — 32 Q&A, `provider=groq` `model=llama-3.1-8b-instant`, **HyDE on** (ADR-011), all on free tier.

| Metric                       | Target    | Pre-HyDE (ADR-010) | **Current** ([ADR-011](docs/DECISIONS.md))  | Source                            |
|------------------------------|-----------|--------------------|----------------------------------------------|-----------------------------------|
| `avg_topic_coverage`         | ≥ 0.85    | 0.526              | **0.563**  (+7 %)                            | `evals/baseline.json`             |
| `avg_citation_match`         | ≥ 0.90    | 0.188              | **0.438**  (+133 %)                          | same                              |
| `citation_rate`              | ≥ 0.95    | 0.312              | **0.562**  (+80 %)                           | same                              |
| **/search P95 latency**      | ≤ 3 s     | -                  | **~3.9 s** (CPU)                             | live HTTP probe via Vercel → HF Space, 5 real queries |
| **rerank stage**             | -         | -                  | **~3.2 s** / query                            | [`scripts/bench_search.py`](scripts/bench_search.py) |
| **dense retrieval**          | -         | -                  | ~0.9 s / query                                | same — Neon network               |
| **sparse + embed + fuse**    | -         | -                  | < 0.3 s / query                               | same                              |
| Avg cost per query           | ≤ $0.005  | $0.00014           | **$0.00013** ✅                              | `baseline.json` (Groq paid-tier pricing applied; actual $ spent: 0) |

Honest read of the table: the eval suite **measured** the bottleneck (retrieval mismatches on conceptual queries pulling language-specific API reference pages), **named** the fix (ADR-011 — HyDE), **quantified** the improvement (+133 % on `citation_match` at flat latency / cost), and the next ADRs (012-014) are the remaining gap to the targets. That's what an eval suite is for.

The "pending API key" rows light up the moment an `ANTHROPIC_API_KEY` is wired —
nothing else needs to change. Latency tuned from ~100 s/query before ADR-009
(`bge-reranker-v2-m3`) to ~3.9 s after (`ms-marco-MiniLM-L-6-v2`).

---

## Architecture (high level)

```
User query
    │
    ▼
┌────────────────────────────────────────┐
│ Hybrid retrieval                       │
│   dense (pgvector HNSW, bge-small)  ─┐ │
│   sparse (bm25s)                    ─┴─► RRF fusion (top-20)
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Cross-encoder reranker                 │
│ (bge-reranker-v2-m3, local)            │
└────────────────────────────────────────┘
    │ top-5
    ▼
┌────────────────────────────────────────┐
│ Claude Messages API                    │
│   - System + CONTEXT (cache breakpoint)│
│   - User question                      │
└────────────────────────────────────────┘
    │
    ▼
Answer with [n] citations + cost + per-stage timings
```

Full details in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Stack

- **Language**: Python 3.12, managed by `uv`
- **LLM**: Anthropic Claude (Haiku 4.5 default, Sonnet 4.6 / Opus 4.7 selectable)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (384-dim, local, fast on CPU — see ADR-008)
- **Sparse retrieval**: `bm25s` (paper 2024, 500× faster than rank_bm25)
- **Reranker**: `BAAI/bge-reranker-v2-m3` (local cross-encoder)
- **Vector store**: Postgres + `pgvector` on Neon serverless (ADR-007)
- **API**: FastAPI + SSE streaming
- **CLI**: Typer (`cdrag` entry point)
- **CI**: GitHub Actions — ruff, mypy --strict, pytest, eval regression gate

---

## Quick start

```powershell
# 1. Install deps (uv handles Python 3.12 + venv)
uv sync

# 2. Configure secrets
cp .env.example .env
# Edit .env and set:
#   POSTGRES_DSN=postgresql://...   (Neon free tier works)
#   ANTHROPIC_API_KEY=sk-ant-...    (only needed for `ask` and `eval`)

# 3. Create the schema in your Postgres
uv run cdrag init-db
uv run cdrag check-db

# 4. Ingest the Anthropic docs corpus (idempotent — safe to re-run)
uv run cdrag ingest --concurrency 8 --pages-per-batch 30

# 5. Build the BM25 index over what is in Postgres
uv run cdrag build-bm25

# 6. Hybrid search (no API key needed)
uv run cdrag search "How do I stream messages from the Claude API?"

# 7. End-to-end answer with citations (needs ANTHROPIC_API_KEY)
uv run cdrag ask "How do I stream messages from the Claude API?"

# 8. Run the eval suite (needs ANTHROPIC_API_KEY)
uv run cdrag eval --limit 5
```

---

## Web UI

A minimal Next.js 16 + React 19 + Tailwind 4 frontend lives under [`web/`](web/).
It hits the FastAPI `/search` endpoint cross-origin (CORS is enabled for
`localhost:3000`) and renders the top-K reranked chunks with their section
breadcrumb, source URL, and content excerpt.

```powershell
# 1. Start the API server in one terminal
uv run cdrag serve --host 127.0.0.1 --port 8000

# 2. Start the Next.js dev server in another
cd web
npm install
npm run dev    # -> http://localhost:3000
```

Configure a non-default backend URL with `NEXT_PUBLIC_API_BASE_URL` before
`npm run dev` (or `npm run build`).

---

## Deploy

The repo is wired for a two-host deploy, both free-tier-friendly:

- **Backend** → Hugging Face Spaces (Docker SDK). Always-on, no card on file,
  2 vCPU + 16 GB RAM. Sync happens automatically via
  [`.github/workflows/sync-to-hf-space.yml`](.github/workflows/sync-to-hf-space.yml)
  whenever `main` moves.
- **Frontend** → Vercel (free Hobby plan; root directory `web/`).
- **Alternative**: a Fly.io path is also wired (`fly.toml` + Dockerfile model
  pre-cache) for anyone wanting a `*.fly.dev` subdomain. ~$2/month.

Step-by-step guide (HF Space creation, GitHub secrets, Vercel import) lives in
[`docs/DEPLOY.md`](docs/DEPLOY.md). The BM25 index bootstraps itself from
Postgres on first server start, so no `data/` directory needs to be shipped.

For a quick local container check:

```bash
docker build -t claude-docs-rag .
docker run -p 8000:8000 \
  -e POSTGRES_DSN=postgresql://... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e CDRAG_CORS_ORIGINS=http://localhost:3000 \
  claude-docs-rag
```

---

## Roadmap

- [x] Scaffolding, infra, ADRs (1–9)
- [x] Ingest pipeline (scraper → chunker → embedder → pgvector), batched + idempotent
- [x] Hybrid retrieval (BM25 + dense + RRF fusion)
- [x] Cross-encoder reranker (`ms-marco-MiniLM`, ADR-009: 100 s → 3.7 s per query)
- [x] Golden eval dataset (32 Q&A across 14 categories — to grow to 100+)
- [x] Eval runner (topic coverage, citation match, latency, cost) writes `latest.json`
- [x] CI workflow scaffolded with regression gate (ruff, mypy --strict, pytest, hadolint)
- [x] Minimal Next.js 16 frontend (`web/`)
- [x] Production deploy + public demo URL — HF Spaces (backend) + Vercel (frontend)
- [x] GitHub Action that mirrors `main` to the HF Space automatically
- [x] **ADR-010** — Pluggable LLM provider (Anthropic + Groq), `LLM_PROVIDER` env var
- [x] First full `cdrag eval` run + numbers committed to [`evals/baseline.json`](evals/baseline.json)
- [x] Activate the eval gate in CI against `evals/baseline.json` ([PR #1 / commit `b144489`](https://github.com/alvarocanoo/claude-docs-rag/pull/1))
- [x] **ADR-011** — HyDE (Hypothetical Document Embeddings) on the dense leg; baseline updated, **+133 % `citation_match`** at flat latency / cost
- [x] **`POST /ask/stream`** SSE endpoint + Next.js `/chat` UI with token streaming, citations panel, and inline cost/latency meta line
- [ ] **ADR-012** — Fix sparse-only fusion drop in `hybrid.py` (real bug, separable from HyDE)
- [ ] **ADR-013** — Promote agent LLM to `llama-3.3-70b-versatile` once dev-day TPD frees up; expected headroom on `citation_rate`
- [ ] Semantic cache (Redis embedding similarity)
- [ ] FastAPI + SSE streaming endpoint on `/ask`
- [ ] Langfuse traces wired

---

## License

MIT — see [`LICENSE`](LICENSE).
