"""MCP (Model Context Protocol) server exposing claude-docs-rag as tools.

A Claude-compatible client (Claude Desktop, the Claude CLI, any agent using
the official MCP SDKs) can call:

  - `search_claude_docs`: hybrid retrieval (BM25 + dense + reranker) over the
    Anthropic Claude API docs. Returns top-K reranked chunks with section,
    URL and excerpt. No LLM call, no API key required.

  - `ask_claude_docs`: full RAG — retrieves and asks the configured LLM
    provider (Groq Llama 3.1 8B by default; settable via LLM_PROVIDER) to
    answer the question with bracket-citation markers. Requires GROQ_API_KEY
    (or ANTHROPIC_API_KEY) on the host running this server.

The server runs on stdio by default; wire it into a Claude Desktop config
file as `claude-docs-rag` and the tools show up in the client. See
`docs/MCP.md` for the exact config snippet.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("claude-docs-rag")


@mcp.tool()
async def search_claude_docs(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Hybrid retrieval over the Anthropic Claude API docs (42k chunks of
    `platform.claude.com/docs`). Returns the top-K chunks ranked by a
    cross-encoder reranker after BM25 + dense fusion.

    Args:
        query: natural-language question, e.g. "How do I stream messages?"
        k: number of results to return (1-20). Default 5.

    Returns one dict per result with `source_url`, `title`, `section_path`,
    `rerank_score`, `fusion_score`, and a short `excerpt` of the chunk.
    """
    # Lazy import keeps the FastMCP startup cheap when the search tool is
    # never called (e.g., a host that only uses `ask_claude_docs`).
    from claude_docs_rag.retrieval.hybrid import hybrid_search

    k_clamped = max(1, min(k, 20))
    hits = await hybrid_search(query, top_k_rerank=k_clamped)
    return [
        {
            "source_url": h.source_url,
            "title": h.title,
            "section_path": h.section_path,
            "rerank_score": float(h.rerank_score),
            "fusion_score": float(h.fusion_score),
            "excerpt": " ".join(h.content.split())[:300],
        }
        for h in hits
    ]


@mcp.tool()
async def ask_claude_docs(question: str, k: int = 5, max_tokens: int = 800) -> dict[str, Any]:
    """End-to-end RAG: retrieve, then ask the configured LLM to answer the
    question grounded in the retrieved context. The answer includes inline
    `[n]` citations pointing at the returned `citations` array.

    Args:
        question: natural-language question.
        k: top-K chunks fed to the LLM (1-20). Default 5.
        max_tokens: cap on the answer length (64-4096). Default 800.

    Returns a dict with `answer`, `citations` (list of {chunk_id, source_url,
    section_path}), `provider`/`model` used, token counts and computed
    `cost_usd`, and per-stage `timings_ms`.
    """
    from claude_docs_rag.agent.pipeline import answer_question
    from claude_docs_rag.settings import settings

    if not settings.is_llm_configured:
        raise RuntimeError(
            f"LLM provider {settings.llm_provider!r} has no credentials. "
            "Set GROQ_API_KEY (or ANTHROPIC_API_KEY) before calling this tool."
        )

    k_clamped = max(1, min(k, 20))
    tokens_clamped = max(64, min(max_tokens, 4096))
    result = await answer_question(question, top_k_rerank=k_clamped, max_tokens=tokens_clamped)
    return {
        "question": result.question,
        "answer": result.answer,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "source_url": c.source_url,
                "section_path": c.section_path,
            }
            for c in result.citations
        ],
        "provider": result.call.provider,
        "model": result.call.model,
        "input_tokens": result.call.input_tokens,
        "output_tokens": result.call.output_tokens,
        "cost_usd": result.call.cost_usd,
        "timings_ms": {k: v * 1000 for k, v in result.timings.items()},
    }


def run() -> None:
    """Entry point used by `cdrag mcp` (stdio transport, the default that
    Claude Desktop and the official MCP CLI expect)."""
    mcp.run()
