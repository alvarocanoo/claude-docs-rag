# Architecture Decision Records

> Every non-obvious technical choice in this repo is recorded here with the alternatives considered and the trade-off accepted. This file is the single most important artifact for a technical interview: it lets a reviewer audit the reasoning, not just the code.

Format inspired by Michael Nygard's ADRs, intentionally lightweight.

---

## ADR-001 — Hybrid retrieval (BM25 + dense + RRF) over pure semantic

**Date**: 2026-05-25
**Status**: Accepted

### Context

Naïve RAG uses only vector similarity. This fails on:
- Acronyms and exact tokens (e.g. `MAX_TOKENS`, `tool_use_id`) — embeddings blur them.
- Recently-added terms that the embedding model never saw during pre-training.
- Numeric / version queries (`Claude 4.7`, `2024-10-22`).

### Decision

Run BM25 and dense retrieval in parallel, fuse top-K with Reciprocal Rank Fusion (RRF, k=60), then rerank.

### Alternatives considered

- **Pure semantic (single retriever)**: simplest, but fails the above query types. Rejected.
- **Pure BM25**: misses paraphrases ("how do I stream" vs "Server-Sent Events"). Rejected.
- **ColBERT / late-interaction**: stronger but ~10× compute and complex storage. Rejected as overkill for ~10k chunks.

### Consequences

- (+) Robust across query types — confirmed by ablation in literature (e.g. RAG-Survey 2024).
- (+) Easy to ablate: can disable either retriever via config to measure contribution.
- (−) Two indexes to maintain (HNSW + BM25). Mitigated: BM25 is small (~MB), rebuilt offline.

### How we'll measure this was right

`evals/ablation.py` will compare {dense only, BM25 only, hybrid} on golden set. If hybrid does not beat both single retrievers by ≥ 3 % faithfulness, revisit.

---

## ADR-002 — Cross-encoder reranker mandatory

**Date**: 2026-05-25
**Status**: Accepted

### Context

Retrieval (sparse + dense) returns top-K with bi-encoder semantics, which under-ranks the most relevant chunk in ~20-40 % of queries (per multiple papers, e.g. Cohere Rerank Eval 2024).

### Decision

Retrieve top-20, rerank with `BAAI/bge-reranker-v2-m3`, pass top-5 to the LLM.

### Alternatives considered

- **No reranker, top-5 from retrieval**: simpler, cheaper, but worse answer grounding.
- **Cohere Rerank 3 (API)**: marginally better but $$ and adds a hard external dependency. Pivotable later via env var.
- **LLM-based reranker (Claude Haiku to rank)**: too slow and too expensive for every query.

### Consequences

- (+) Better grounding → higher faithfulness.
- (+) Local model = zero per-query cost, no external dependency.
- (−) Adds ~150-300 ms latency. Acceptable: budget allows up to 3 s P95.

### How we'll measure

Ablation `{no rerank, local rerank, Cohere rerank}` on golden set. Local rerank must beat "no rerank" by ≥ 2 % faithfulness to justify keeping it.

---

## ADR-003 — Postgres + pgvector over dedicated vector DB

**Date**: 2026-05-25
**Status**: Accepted

### Context

Need to store ~10k document chunks with 1024-dim embeddings and run cosine-similarity search at < 100 ms.

### Decision

Postgres 17 + `pgvector` with HNSW index.

### Alternatives considered

- **Pinecone**: managed, fast, but vendor lock-in, monthly cost even at low scale, and no transactional guarantees with the rest of app data.
- **Qdrant / Weaviate / Milvus**: open source, good performance, but yet another service to operate.
- **ChromaDB**: easy local, but limited operational story for production.

### Consequences

- (+) Single database — joins with metadata tables, transactional ingest.
- (+) Ops cost = 1 Postgres instance. Already required for application data.
- (+) Universal: any company hiring will know Postgres.
- (−) pgvector HNSW is competitive but not the fastest at extreme scale (≥ 100M vectors). Not a concern here.

### How we'll measure

Benchmark: ingest 10k chunks, run 1000 random queries. Target P95 < 100 ms on a `pg17` Docker container with 2 GB RAM.

---

## ADR-004 — Eval suite with CI regression gate

**Date**: 2026-05-25
**Status**: Accepted

### Context

The single most common RAG failure mode is silent quality regression: someone changes a prompt or chunk size, faithfulness drops 8 %, nobody notices for weeks.

### Decision

- Maintain `evals/golden_dataset.jsonl` (≥ 100 Q&A with expected citation IDs).
- `pytest tests/evals/` runs the full pipeline and computes faithfulness, citation accuracy, latency, cost.
- CI runs evals on every PR. Compare to baseline in `evals/baseline.json`. Block merge if any metric regresses > 2 %.
- On merge to `main`, baseline is updated to the new measurements.

### Alternatives considered

- **No evals**: industry default for portfolio projects. Why this is wrong: it makes every change a guess.
- **Manual eval before release**: doesn't scale, easy to skip.
- **External tool (Braintrust / LangSmith)**: good products, but adds dependency and cost. Local pytest is enough for this scale.

### Consequences

- (+) Quality is monitored, not assumed.
- (+) The CI badge in README is a strong signal to reviewers.
- (−) Evals cost real Anthropic tokens on every PR. Mitigated: small golden set + prompt caching of the static system prompt.

---

## ADR-005 — `uv` over `pip` / `poetry` / `pdm`

**Date**: 2026-05-25
**Status**: Accepted

### Context

Python dependency management options have proliferated. Picking one affects every contributor and the CI.

### Decision

`uv` from Astral.

### Alternatives considered

- **pip + venv**: ubiquitous but slow, no lockfile, no resolver guarantees.
- **Poetry**: lockfile and resolver, but slow installs and history of breaking changes.
- **PDM**: PEP 582 friendly, less momentum.
- **uv**: 10–100× faster than pip, drop-in resolver, single binary, by the Ruff team.

### Consequences

- (+) CI installs go from minutes to seconds.
- (+) Lockfile (`uv.lock`) is deterministic.
- (−) Slightly newer tool — minor docs friction for contributors who haven't used it. Mitigated by README quick start.
