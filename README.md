# claude-docs-rag

> Production-grade RAG agent over the Anthropic Claude API documentation. Hybrid retrieval (BM25 + embeddings + reranking), eval suite with CI regression gates, observability, semantic caching and multi-model routing.

**Status**: WIP — scaffolding phase. Not yet usable.

---

## Why this exists

Most public RAG demos are toys: single retriever, no evals, no observability, no caching, no cost discipline. This repo is the opposite — a small but **defensible** system where every architectural choice has a documented trade-off and a measurable result.

If you're an engineer reviewing this for hiring purposes, the most interesting files are likely:

- [`docs/DECISIONS.md`](docs/DECISIONS.md) — Architecture Decision Records with trade-offs.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System diagram and request flow.
- `evals/` — Golden dataset + scoring logic. Drives the CI regression gate.
- `.github/workflows/ci.yml` — CI gate that blocks merges if quality drops > 2%.

---

## Success metrics (declared up-front)

These are the numerical targets the system must hit before being considered "done":

| Metric                       | Target            | How it's measured                          |
|------------------------------|-------------------|--------------------------------------------|
| Faithfulness                 | ≥ 0.85            | LLM-as-judge on golden eval set            |
| Citation accuracy            | ≥ 0.90            | Cited chunk must match the answer claim    |
| End-to-end P95 latency       | ≤ 3 s             | N=100 real queries, traced via Langfuse    |
| Avg cost per query (Haiku)   | ≤ $0.005          | Token accounting from Anthropic responses  |
| Semantic cache hit rate      | ≥ 30 %            | On eval set with paraphrased queries       |

Current measured values: **not yet measured** — first eval run will populate this table.

---

## Architecture (high level)

```
User query
    │
    ▼
┌────────────────┐
│ Semantic cache │ ──── hit ────► cached answer
└────────────────┘
    │ miss
    ▼
┌────────────────────────────────────────┐
│ Hybrid retrieval (BM25 + dense + RRF)  │
└────────────────────────────────────────┘
    │ top-20
    ▼
┌────────────────┐
│ Cross-encoder  │
│ reranker       │
└────────────────┘
    │ top-5
    ▼
┌────────────────────────────────────────┐
│ Model router (Haiku / Sonnet / local)  │
└────────────────────────────────────────┘
    │
    ▼
Answer + citations (streamed via SSE)
```

Full details in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Stack

- **Language**: Python 3.12
- **LLM**: Anthropic Claude (Haiku 4.5 + Sonnet 4.6, with local Ollama fallback)
- **Embeddings**: `BAAI/bge-m3` (local, multilingual)
- **Sparse retrieval**: `bm25s` (state-of-the-art BM25, paper 2024)
- **Reranker**: `BAAI/bge-reranker-v2-m3` (local cross-encoder)
- **Vector store**: Postgres 17 + `pgvector`
- **Cache / queue**: Redis 8
- **API**: FastAPI + Server-Sent Events
- **Observability**: Langfuse (self-hosted)
- **Package management**: `uv`
- **CI**: GitHub Actions with eval regression gate

Decisions justified in [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Quick start (when ready)

```bash
# 1. Start infra
docker compose up -d

# 2. Install deps
uv sync

# 3. Configure secrets
cp .env.example .env
# fill ANTHROPIC_API_KEY at minimum

# 4. Ingest the docs
uv run python -m claude_docs_rag.ingest.run

# 5. Run the API
uv run uvicorn claude_docs_rag.api.server:app --reload

# 6. Run evals
uv run pytest tests/evals
```

---

## Roadmap

- [x] Scaffolding, infra, ADRs
- [ ] Ingest pipeline (scraper → chunker → embedder → pgvector)
- [ ] Hybrid retrieval (BM25 + dense + RRF fusion)
- [ ] Cross-encoder reranker
- [ ] Golden eval dataset (≥ 100 Q&A)
- [ ] Eval suite (faithfulness, citation accuracy, latency, cost)
- [ ] CI regression gate
- [ ] Semantic cache + cost telemetry
- [ ] Multi-model routing (Haiku / Sonnet / local)
- [ ] FastAPI + SSE streaming
- [ ] Langfuse traces wired
- [ ] Minimal Next.js frontend
- [ ] Production deploy (Fly.io or Railway) + public demo URL

---

## License

MIT — see [`LICENSE`](LICENSE).
