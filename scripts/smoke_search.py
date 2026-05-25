"""Smoke search — embed a query and dump the top-K retrieved chunks."""

from __future__ import annotations

import asyncio
import sys

from claude_docs_rag.ingest.embedder import embed_texts
from claude_docs_rag.storage.vector_store import search_semantic

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


QUERIES = [
    "How do I stream messages from the Claude API?",
    "What is the maximum context window of Claude Sonnet?",
    "How do I use tool_use with the Messages API?",
    "Show me how to define a tool schema for Claude",
    "How do I count input tokens before sending a request?",
]


async def main() -> None:
    for q in QUERIES:
        vec = embed_texts([q])[0]
        results = await search_semantic(vec, k=3)
        print(f"\n=== Q: {q}")
        if not results:
            print("  (no results — DB empty?)")
            continue
        for i, r in enumerate(results, 1):
            head = r.section_path[:80] if r.section_path else r.title
            print(f"  {i}. [score={r.score:.3f}] {head}")
            print(f"     url:    {r.source_url}")
            preview = " ".join(r.content.split())[:160]
            print(f"     excerpt: {preview}...")


if __name__ == "__main__":
    asyncio.run(main())
