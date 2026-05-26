"""Hybrid retrieval: dense (pgvector) + sparse (bm25s) fused with RRF, then reranked."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from claude_docs_rag.ingest.embedder import embed_texts
from claude_docs_rag.retrieval import sparse
from claude_docs_rag.retrieval.fusion import reciprocal_rank_fusion
from claude_docs_rag.retrieval.hyde import expand_for_dense
from claude_docs_rag.retrieval.reranker import rerank
from claude_docs_rag.settings import settings
from claude_docs_rag.storage.vector_store import RetrievedChunk, search_semantic


class HybridResult(BaseModel):
    source_url: str
    title: str
    section_path: str
    chunk_index: int
    content: str
    rerank_score: float
    fusion_score: float


async def hybrid_search(
    query: str,
    *,
    top_k_retrieval: int | None = None,
    top_k_rerank: int | None = None,
) -> list[HybridResult]:
    top_k_retrieval = top_k_retrieval or settings.top_k_retrieval
    top_k_rerank = top_k_rerank or settings.top_k_rerank

    # HyDE (ADR-011): optionally expand the query with an LLM-generated
    # hypothetical passage before embedding. BM25 always sees the raw query
    # because it's keyword-driven.
    dense_query_text = await expand_for_dense(query)

    # Run dense + sparse in parallel.
    query_vec = embed_texts([dense_query_text])[0]
    dense_task = asyncio.create_task(search_semantic(query_vec, k=top_k_retrieval))
    sparse_hits = await asyncio.to_thread(sparse.search, query, k=top_k_retrieval)
    dense_hits = await dense_task

    dense_ranked = [(h.source_url, h.chunk_index) for h in dense_hits]
    sparse_ranked = [(h.source_url, h.chunk_index) for h in sparse_hits]
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], top_k=top_k_retrieval)

    # Map back to the full chunk row (only dense_hits carry the content; for
    # candidates returned only by BM25 we'd need a DB lookup — kept simple here
    # by retaining only the dense-side payloads. A second SELECT can be added
    # if we observe a high fraction of "sparse-only" winners.)
    by_key: dict[tuple[str, int], RetrievedChunk] = {
        (h.source_url, h.chunk_index): h for h in dense_hits
    }
    candidates: list[tuple[RetrievedChunk, float]] = []
    for item in fused:
        chunk = by_key.get((item.source_url, item.chunk_index))
        if chunk is not None:
            candidates.append((chunk, item.score))

    if not candidates:
        return []

    rerank_input = [c[0].content for c in candidates]
    reranked = await asyncio.to_thread(rerank, query, rerank_input, top_k=top_k_rerank)

    return [
        HybridResult(
            source_url=candidates[orig_idx][0].source_url,
            title=candidates[orig_idx][0].title,
            section_path=candidates[orig_idx][0].section_path,
            chunk_index=candidates[orig_idx][0].chunk_index,
            content=candidates[orig_idx][0].content,
            rerank_score=score,
            fusion_score=candidates[orig_idx][1],
        )
        for orig_idx, score in reranked
    ]
