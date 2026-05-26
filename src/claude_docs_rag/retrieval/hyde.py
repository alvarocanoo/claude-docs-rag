"""HyDE (Hypothetical Document Embeddings) — `arxiv 2212.10496`.

Idea: a user question and a documentation passage live in different
distributions in embedding space ("how do I X?" vs. "X works by..."). By
asking the LLM to *first answer* the question (even imperfectly), we
embed something that looks like a doc passage and match the corpus better.

Wire-up:
  1. `cdrag eval` / `/search` calls `hybrid_search(query, ...)`.
  2. If `settings.hyde_enabled`, the dense leg embeds `_hyde_expand(query)`
     instead of the raw query. BM25 still runs on the raw query because
     BM25 is keyword-driven and benefits from the original verbatim terms.

Failure modes are handled defensively: if the LLM call errors or times out,
we fall back to the raw query so retrieval still works.

See ADR-011 for the ablation against `evals/baseline.json`.
"""

from __future__ import annotations

import asyncio

from claude_docs_rag.agent.client import create_message
from claude_docs_rag.settings import settings

# Kept short on purpose: the goal is to produce *embeddable* prose, not a
# polished answer. ~150 tokens is plenty.
SYSTEM_PROMPT = (
    "You write a short, plausible passage of Anthropic Claude API documentation "
    "that would answer the user's question. Write in the voice of the docs: "
    "explain the mechanism, name the relevant parameters or endpoints, and "
    "include a tiny code or JSON sketch if it fits. Do not say 'I think' or "
    "speculate; write as if you were the doc itself. 4-6 sentences max."
)
HYDE_MAX_TOKENS = 200


async def _hyde_expand(query: str) -> str:
    """Return `query + "\\n\\n" + hypothetical_passage` if HyDE is enabled and
    the LLM call succeeds; otherwise return the raw `query`."""
    if not settings.hyde_enabled or not settings.is_llm_configured:
        return query
    try:
        result = await asyncio.wait_for(
            create_message(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
                max_tokens=HYDE_MAX_TOKENS,
                temperature=0.0,
            ),
            timeout=settings.hyde_timeout_seconds,
        )
        passage = (result.text or "").strip()
        if not passage:
            return query
        return f"{query}\n\n{passage}"
    except Exception:
        # Defensive: never let HyDE break retrieval. Fall back to the raw query.
        return query


async def expand_for_dense(query: str) -> str:
    """Public entry point: returns the string to embed for dense retrieval."""
    return await _hyde_expand(query)
