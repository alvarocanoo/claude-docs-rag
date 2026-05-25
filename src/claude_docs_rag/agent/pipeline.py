"""End-to-end RAG pipeline: hybrid retrieval -> Claude answer with citations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from anthropic.types import MessageParam, TextBlockParam

from claude_docs_rag.agent.client import CallResult, create_message
from claude_docs_rag.retrieval.hybrid import HybridResult, hybrid_search
from claude_docs_rag.settings import settings

SYSTEM_PROMPT = """You answer developer questions about the Anthropic Claude API
strictly from the CONTEXT block. Each CONTEXT chunk is preceded by an integer id
in brackets, e.g. [3]. Cite the id(s) that support each non-trivial claim using
the same bracket notation inline (e.g. "Streaming is enabled via... [3][5]").

Rules:
- If the CONTEXT does not contain the answer, say so and do NOT invent facts.
- Prefer short answers. Include code only when the user asks for it or the docs
  show it.
- Always include at least one citation when the answer is grounded.
"""


@dataclass
class Citation:
    chunk_id: int
    source_url: str
    section_path: str


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[HybridResult]
    call: CallResult
    latency_seconds: float
    timings: dict[str, float] = field(default_factory=dict)


def _build_context(chunks: list[HybridResult]) -> str:
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({ch.section_path})\n{ch.content}\n")
    return "\n".join(parts)


def _extract_citation_ids(answer_text: str, max_id: int) -> list[int]:
    """Pull integer ids found in [n] form, dedup-preserve order, ≤ max_id."""
    seen: list[int] = []
    i = 0
    while i < len(answer_text):
        if answer_text[i] == "[":
            j = i + 1
            while j < len(answer_text) and answer_text[j].isdigit():
                j += 1
            if j > i + 1 and j < len(answer_text) and answer_text[j] == "]":
                n = int(answer_text[i + 1 : j])
                if 1 <= n <= max_id and n not in seen:
                    seen.append(n)
                i = j + 1
                continue
        i += 1
    return seen


async def answer_question(
    question: str,
    *,
    model: str | None = None,
    top_k_rerank: int | None = None,
    max_tokens: int = 1024,
) -> RagAnswer:
    started = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    retrieved = await hybrid_search(question, top_k_rerank=top_k_rerank)
    timings["retrieval"] = time.perf_counter() - t0

    if not retrieved:
        return RagAnswer(
            question=question,
            answer="No relevant context was retrieved. Cannot answer from sources.",
            citations=[],
            retrieved=[],
            call=CallResult(
                text="",
                model=model or settings.model_simple,
                input_tokens=0,
                output_tokens=0,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                stop_reason=None,
                cost_usd=0.0,
            ),
            latency_seconds=time.perf_counter() - started,
            timings=timings,
        )

    context_block = _build_context(retrieved)

    # Cache the system + context as one breakpoint. The system prompt is static;
    # we still attach cache_control to the context so prompt caching kicks in
    # when the same context is reused for a follow-up question in the session.
    system_blocks: list[TextBlockParam] = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {
            "type": "text",
            "text": f"CONTEXT (cite by id in brackets):\n{context_block}",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    messages: list[MessageParam] = [{"role": "user", "content": question}]

    t0 = time.perf_counter()
    call = await create_message(
        model=model or settings.model_simple,
        system=system_blocks,
        messages=messages,
        max_tokens=max_tokens,
    )
    timings["generation"] = time.perf_counter() - t0

    cited_ids = _extract_citation_ids(call.text, max_id=len(retrieved))
    citations = [
        Citation(
            chunk_id=cid,
            source_url=retrieved[cid - 1].source_url,
            section_path=retrieved[cid - 1].section_path,
        )
        for cid in cited_ids
    ]

    return RagAnswer(
        question=question,
        answer=call.text,
        citations=citations,
        retrieved=retrieved,
        call=call,
        latency_seconds=time.perf_counter() - started,
        timings=timings,
    )
