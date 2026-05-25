# Architecture

> System design for `claude-docs-rag`. Decisions are recorded in [`DECISIONS.md`](DECISIONS.md).

## Goal

Given a developer's question about the Anthropic Claude API, return a grounded answer with verifiable citations, faster than scrolling docs by hand, at <$0.005 per query.

## Request flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                            Client                                    │
│                       (Next.js, curl, etc.)                          │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ POST /chat (SSE)
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI gateway                                │
│  - Auth (API key)                                                    │
│  - Rate limit (Redis token bucket)                                   │
│  - Request validation (pydantic)                                     │
│  - Langfuse trace open                                               │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Semantic cache check                            │
│   - Embed query → Redis vector similarity search                    │
│   - If cos_sim ≥ 0.93 → return cached answer                        │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ miss
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Hybrid retrieval                               │
│   ┌──────────────┐    ┌──────────────┐                              │
│   │ BM25 (bm25s) │    │ Dense (bge)  │                              │
│   └──────┬───────┘    └──────┬───────┘                              │
│          │     top-20        │ top-20                               │
│          └────────┬──────────┘                                      │
│                   ▼                                                  │
│              RRF fusion → top-20                                    │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                Cross-encoder reranker (bge-reranker-v2-m3)           │
│                       top-20 → top-5                                 │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Model router                                       │
│   - Heuristic: short factual → Haiku 4.5                            │
│   - Heuristic: code generation / multi-step → Sonnet 4.6            │
│   - On Anthropic 5xx → local Ollama (qwen2.5:7b)                    │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│             Anthropic Messages API (with prompt cache)               │
│   - System prompt + retrieved chunks (cache breakpoint)              │
│   - User question                                                    │
│   - Streamed response                                                │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ SSE
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│   Response post-processing                                           │
│   - Parse citations [doc:X:line:Y]                                   │
│   - Write to Langfuse (tokens, cost, latency)                       │
│   - Write to semantic cache (Redis)                                  │
└─────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
                Client
```

## Storage layout (pgvector)

```sql
CREATE TABLE documents (
    id              BIGSERIAL PRIMARY KEY,
    source_url      TEXT NOT NULL,
    title           TEXT NOT NULL,
    section_path    TEXT NOT NULL,         -- e.g. "Build / Messages / Streaming"
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    content_tokens  INT NOT NULL,
    embedding       VECTOR(1024),          -- bge-m3 dim
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_url, chunk_index)
);

CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON documents USING gin (to_tsvector('english', content));
```

## Ingestion pipeline

1. **Scrape**: download docs (sitemap or git clone of Anthropic docs repo).
2. **Parse**: extract markdown structure (headings, code blocks).
3. **Chunk**: header-aware splitter, target ~600 tokens, 100-token overlap.
4. **Embed**: `bge-m3`, batch size 32, cached on disk.
5. **Store**: insert to pgvector, build BM25 index (`bm25s` artifact in Redis).

## Eval loop

Every PR runs `pytest tests/evals/` which:
1. Loads `evals/golden_dataset.jsonl` (≥ 100 Q&A pairs with expected answer + expected citations).
2. Runs each query through the live pipeline (test Postgres + test Redis).
3. Computes faithfulness (LLM-as-judge), citation accuracy (set overlap), latency P95, cost.
4. Compares to baseline stored in `evals/baseline.json` (updated on merge to main).
5. Fails the build if any metric regresses > 2 % from baseline.

## Observability

- Langfuse traces every step (cache lookup, BM25, dense, fusion, rerank, model call).
- Tags: `model`, `cache_hit`, `route`.
- Dashboards: cost/day, P95 latency, hit rate, faithfulness trend.

## Deploy target

- **Local dev**: `docker compose up -d` for infra, `uv run uvicorn ...` for app.
- **Production**: Fly.io or Railway. Postgres as managed (Supabase or Neon) or Fly Postgres. Redis as Upstash. Langfuse stays self-hosted on the same host.

## Non-goals (deliberately out of scope)

- Multi-tenant support / user accounts.
- Fine-tuning (the docs change too fast to be worth it).
- General-purpose chat (this is doc QA only — refuses off-topic queries).
