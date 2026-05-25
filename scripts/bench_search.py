"""Per-stage timing for hybrid_search — identifies the latency bottleneck."""

from __future__ import annotations

import asyncio
import sys
import time

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


async def main() -> None:
    from claude_docs_rag.ingest.embedder import embed_texts
    from claude_docs_rag.retrieval import sparse
    from claude_docs_rag.retrieval.fusion import reciprocal_rank_fusion
    from claude_docs_rag.retrieval.reranker import rerank
    from claude_docs_rag.storage.vector_store import search_semantic

    queries = [
        "How do I stream messages from the Claude API?",
        "How does prompt caching work?",
        "What is the bash tool?",
    ]

    # Warmup once (load every cached resource).
    print("Warming up...")
    t0 = time.perf_counter()
    _ = embed_texts(["warmup"])
    print(f"  embedder cold: {(time.perf_counter() - t0) * 1000:.0f} ms")

    t0 = time.perf_counter()
    _ = sparse.search("warmup", k=5)
    print(f"  sparse cold:   {(time.perf_counter() - t0) * 1000:.0f} ms")

    t0 = time.perf_counter()
    vec = embed_texts(["warmup"])[0]
    _ = await search_semantic(vec, k=5)
    print(f"  dense cold:    {(time.perf_counter() - t0) * 1000:.0f} ms")

    t0 = time.perf_counter()
    _ = rerank("warmup", ["passage one", "passage two"])
    print(f"  rerank cold:   {(time.perf_counter() - t0) * 1000:.0f} ms")

    print("\n=== Warm timings, 20 candidates -> rerank top-5 ===")
    for q in queries:
        print(f"\nQ: {q}")
        t = time.perf_counter()
        vec = embed_texts([q])[0]
        t_embed = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        dense_hits = await search_semantic(vec, k=20)
        t_dense = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        sparse_hits = sparse.search(q, k=20)
        t_sparse = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        dense_keys = [(h.source_url, h.chunk_index) for h in dense_hits]
        sparse_keys = [(h.source_url, h.chunk_index) for h in sparse_hits]
        fused = reciprocal_rank_fusion([dense_keys, sparse_keys], top_k=20)
        t_fuse = (time.perf_counter() - t) * 1000

        by_key = {(h.source_url, h.chunk_index): h for h in dense_hits}
        passages = [
            by_key[(item.source_url, item.chunk_index)].content
            for item in fused
            if (item.source_url, item.chunk_index) in by_key
        ]

        t = time.perf_counter()
        _ = rerank(q, passages, top_k=5)
        t_rerank = (time.perf_counter() - t) * 1000

        total = t_embed + t_dense + t_sparse + t_fuse + t_rerank
        print(
            f"  embed={t_embed:6.0f}ms  dense={t_dense:6.0f}ms  sparse={t_sparse:6.0f}ms  "
            f"fuse={t_fuse:5.1f}ms  rerank={t_rerank:7.0f}ms ({len(passages)} passages)  "
            f"TOTAL={total:7.0f}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
