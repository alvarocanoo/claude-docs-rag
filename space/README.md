---
title: claude-docs-rag
emoji: "\U0001F50D"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Hybrid RAG over the Anthropic Claude API docs.
---

# claude-docs-rag — Hugging Face Space

Production-grade RAG agent over the Anthropic Claude API docs. Hybrid retrieval
(BM25 + `BAAI/bge-small-en-v1.5` embeddings on `pgvector` HNSW + RRF fusion)
with a `ms-marco-MiniLM-L-6-v2` cross-encoder reranker. FastAPI backend +
Next.js 16 frontend (frontend hosted separately).

## API

Once the Space is running, hit the JSON API directly:

```bash
# Health
curl https://<user>-claude-docs-rag.hf.space/healthz

# Hybrid retrieval (no API key needed)
curl -X POST https://<user>-claude-docs-rag.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I stream messages from the Claude API?","k":5}'
```

## Source code and full documentation

GitHub: https://github.com/alvarocanoo/claude-docs-rag — `README.md`,
`docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/DEPLOY.md`.

This Space is built and deployed automatically from `main` via a GitHub Action
(`.github/workflows/sync-to-hf-space.yml`).
