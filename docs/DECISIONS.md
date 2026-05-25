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

Postgres 17 + `pgvector` with HNSW index. See ADR-007 for hosting choice (Neon vs local).

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

Benchmark: ingest 10k chunks, run 1000 random queries. Target P95 < 100 ms.

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

---

## ADR-006 — Anthropic docs source: `llms.txt` index + per-page `.md`

**Date**: 2026-05-25
**Status**: Accepted

### Context

Need to ingest the Anthropic Claude API docs. Options range from HTML scraping (fragile, slow, needs JS rendering) to a single bulk file.

Anthropic publishes both:
- `https://platform.claude.com/llms.txt` — 166 KB index listing 1541 English doc pages by URL.
- `https://platform.claude.com/llms-full.txt` — 66 MB monolithic markdown of all pages.
- A `.md` endpoint per individual page (e.g. `/docs/en/intro.md`).

### Decision

Use `llms.txt` as the index and download each page individually via its `.md` endpoint, with bounded concurrency and on-disk caching.

### Alternatives considered

- **Scrape HTML**: needs Playwright for SPA content, slow (~1 s/page), fragile to UI changes. Rejected.
- **Bulk download `llms-full.txt`**: 1 HTTP request, simple — but the monolith has no clean per-page boundaries, complicating metadata (URL, section) and per-page caching/retry. Rejected.
- **Per-page `.md` from `llms.txt` (chosen)**: each page becomes a unit with stable URL, title and section. Granular caching, retry, idempotent re-ingest.

### Consequences

- (+) Metadata-rich: every chunk knows its source URL and section path.
- (+) Robust to partial failures: re-running only re-downloads missing files.
- (+) English-only filter at index level keeps the corpus focused and consistent.
- (−) 1541 HTTP requests. Mitigated: concurrency 10, on-disk cache so it only happens once.

---

## ADR-007 — Neon serverless Postgres for dev and prod

**Date**: 2026-05-25
**Status**: Accepted

### Context

ADR-003 picked pgvector. Hosting it was originally docker-compose locally + a managed Postgres in prod (Supabase/Neon/Fly Postgres). Reality on the dev box: no admin privileges → cannot install WSL2 → Docker Desktop won't start its Linux engine.

### Decision

Use **Neon serverless Postgres** for both development and production, accessed via a single `POSTGRES_DSN` connection string in `.env`.

### Alternatives considered

- **Local Postgres native install**: needs admin for `winget install PostgreSQL.PostgreSQL.17` and a separate pgvector binary. Operationally heavier.
- **WSL2 + Docker**: requires admin + system reboot.
- **SQLite + sqlite-vec**: drops pgvector entirely, breaks ADR-003 and reduces portfolio signal.
- **Neon free tier (chosen)**: zero local install, pgvector preinstalled, 0.5 GB free is ample (~10k chunks × 1024 floats ≈ 40 MB), database branching for per-PR eval isolation in CI.

### Consequences

- (+) Same backend in dev, CI, prod. No "works on my machine".
- (+) Database branching: each PR can spin up an isolated branch from `main` data → evals run against real data without polluting it.
- (+) Connection pooler included → safe for serverless workers.
- (−) Requires internet for dev. Acceptable for this project.
- (−) Cold-start latency on free tier (~0.5-2 s after idle). Mitigated for evals by keep-alive query at CI start.

### How we'll measure

Cost: must stay inside free tier through development. Latency: P95 retrieval still ≤ 100 ms once warm.

---

## ADR-008 — Switch embedding model from `bge-m3` to `bge-small-en-v1.5`

**Date**: 2026-05-25
**Status**: Accepted (supersedes embedding choice in ADR-001)

### Context

ADR-001 picked `BAAI/bge-m3` (1024-dim, multilingual). On the dev machine (CPU only, no admin, no GPU) embedding the full 72486 chunks of the Anthropic docs corpus did not complete in 50+ minutes of wall-clock. The corpus is English-only, so the multilingual capacity of bge-m3 is unused weight.

### Decision

Use `BAAI/bge-small-en-v1.5` — 384-dim, English-only, ~5× faster on CPU. Update `EMBEDDING_DIM` in the embedder and `vector(384)` in the schema.

### Alternatives considered

- **bge-m3** (status quo): too slow on CPU for this corpus.
- **bge-base-en-v1.5** (768-dim): middle ground, ~3× faster than m3.
- **OpenAI text-embedding-3-small** (1536-dim, hosted): trivial cost (~$0.002 for full corpus) and ~10× faster than local CPU, but adds an API dependency, account, and key handling.
- **bge-small-en-v1.5 (chosen)**: zero external dependency, runs in seconds per batch, MTEB English benchmarks within ~2 % of bge-m3 for retrieval. Acceptable for this use case.

### Consequences

- (+) Ingest fits in minutes, not hours, on the same hardware.
- (+) Smaller index footprint in pgvector (384 floats × 4 bytes = 1.5 KB/row vs 4 KB/row).
- (+) Reranker (`bge-reranker-v2-m3`) is unchanged, so final relevance after rerank is largely preserved.
- (−) English-only corpus assumption is now hard-coded. Re-ingesting non-English content would require switching back to a multilingual model.

### How we'll measure

After re-ingest: report full-corpus elapsed time. Run the eval suite and compare `avg_topic_coverage` / `avg_citation_match` against pre-switch baselines — drop must be < 5 % to keep the choice.

