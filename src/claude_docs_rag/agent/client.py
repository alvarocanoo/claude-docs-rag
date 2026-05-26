"""Provider-pluggable async LLM client with cost accounting + retries.

Supported providers (selected via `LLM_PROVIDER` env / `settings.llm_provider`):
  - "anthropic": official Anthropic SDK, supports prompt caching.
  - "groq": official Groq SDK, free-tier-friendly. No prompt caching support.

The public surface is `create_message(...)`; callers (pipeline.py, server.py)
do not need to know which backend is in use.

See `docs/DECISIONS.md` ADR-010 for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, TextBlockParam
from groq import AsyncGroq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from claude_docs_rag.settings import settings

# Anthropic public pricing as of 2026-05 (USD per 1M tokens).
# Source: https://www.anthropic.com/pricing  — verify before any cost-sensitive claim.
PRICING_ANTHROPIC: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
}

# Groq public pricing per 1M tokens — verify at https://groq.com/pricing
# Note: the free tier covers portfolio-scale traffic; these numbers are the
# paid-tier prices used to estimate cost even when no money actually changes
# hands. Update if Groq retiers pricing.
PRICING_GROQ: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}


@dataclass
class CallResult:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    stop_reason: str | None
    cost_usd: float


@lru_cache(maxsize=1)
def _anthropic_client() -> AsyncAnthropic:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is empty — set it in .env before calling the agent.")
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


@lru_cache(maxsize=1)
def _groq_client() -> AsyncGroq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is empty — set it in .env before calling the agent.")
    return AsyncGroq(api_key=settings.groq_api_key)


def _cost_anthropic(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> float:
    prices = PRICING_ANTHROPIC.get(model)
    if not prices:
        return 0.0
    return (
        input_tokens * prices["input"]
        + output_tokens * prices["output"]
        + cache_creation_tokens * prices.get("cache_write", 0.0)
        + cache_read_tokens * prices.get("cache_read", 0.0)
    ) / 1_000_000


def _cost_groq(*, model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING_GROQ.get(model)
    if not prices:
        return 0.0
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def _flatten_system_to_text(system: list[TextBlockParam] | str) -> str:
    """Collapse Anthropic-style system blocks to a single string for providers
    that lack structured system arrays / prompt caching."""
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        text = block.get("text") if isinstance(block, dict) else None
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _resolve_provider(provider: str | None) -> str:
    p = (provider or settings.llm_provider).lower()
    if p not in {"anthropic", "groq"}:
        raise ValueError(f"Unknown LLM provider {p!r}. Set LLM_PROVIDER to 'anthropic' or 'groq'.")
    return p


def _resolve_model(provider: str, model: str | None) -> str:
    if model:
        return model
    if provider == "anthropic":
        return settings.model_simple
    return settings.groq_model


# ---------- Anthropic backend ------------------------------------------------


async def _create_message_anthropic(
    *,
    model: str,
    system: list[TextBlockParam] | str,
    messages: list[MessageParam],
    max_tokens: int,
    temperature: float,
) -> CallResult:
    response: Message = await _anthropic_client().messages.create(
        model=model,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    usage = response.usage
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    return CallResult(
        text=text,
        provider="anthropic",
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_tokens=cache_create,
        cache_read_tokens=cache_read,
        stop_reason=response.stop_reason,
        cost_usd=_cost_anthropic(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=cache_create,
            cache_read_tokens=cache_read,
        ),
    )


# ---------- Groq backend (OpenAI-compatible) --------------------------------


async def _create_message_groq(
    *,
    model: str,
    system: list[TextBlockParam] | str,
    messages: list[MessageParam],
    max_tokens: int,
    temperature: float,
) -> CallResult:
    system_text = _flatten_system_to_text(system)

    chat_messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            # Anthropic content blocks -> flatten to text for Groq.
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        chat_messages.append({"role": m["role"], "content": content})

    response = await _groq_client().chat.completions.create(
        model=model,
        messages=chat_messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = response.usage
    input_tok = usage.prompt_tokens if usage else 0
    output_tok = usage.completion_tokens if usage else 0

    return CallResult(
        text=text,
        provider="groq",
        model=model,
        input_tokens=input_tok,
        output_tokens=output_tok,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        stop_reason=choice.finish_reason,
        cost_usd=_cost_groq(model=model, input_tokens=input_tok, output_tokens=output_tok),
    )


# ---------- Public API ------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def create_message(
    *,
    model: str | None = None,
    system: list[TextBlockParam] | str,
    messages: list[MessageParam],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    provider: str | None = None,
) -> CallResult:
    """Provider-agnostic message creation. Dispatches to the configured backend."""
    p = _resolve_provider(provider)
    m = _resolve_model(p, model)
    if p == "groq":
        return await _create_message_groq(
            model=m,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return await _create_message_anthropic(
        model=m,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
