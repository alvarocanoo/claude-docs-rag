"""Thin async wrapper around the Anthropic SDK with cost accounting + retries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, TextBlockParam
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from claude_docs_rag.settings import settings

# Anthropic public pricing as of 2026-05 (USD per 1M tokens).
# Source: https://www.anthropic.com/pricing  — verify before any cost-sensitive claim.
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
}


@dataclass
class CallResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    stop_reason: str | None
    cost_usd: float


@lru_cache(maxsize=1)
def _client() -> AsyncAnthropic:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is empty — set it in .env before calling the agent.")
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


def _cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> float:
    prices = PRICING.get(model)
    if not prices:
        return 0.0
    return (
        input_tokens * prices["input"]
        + output_tokens * prices["output"]
        + cache_creation_tokens * prices.get("cache_write", 0.0)
        + cache_read_tokens * prices.get("cache_read", 0.0)
    ) / 1_000_000


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def create_message(
    *,
    model: str,
    system: list[TextBlockParam] | str,
    messages: list[MessageParam],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> CallResult:
    response: Message = await _client().messages.create(
        model=model,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    usage = response.usage
    input_tok = usage.input_tokens
    output_tok = usage.output_tokens
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    return CallResult(
        text=text,
        model=model,
        input_tokens=input_tok,
        output_tokens=output_tok,
        cache_creation_tokens=cache_create,
        cache_read_tokens=cache_read,
        stop_reason=response.stop_reason,
        cost_usd=_cost_usd(
            model=model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cache_creation_tokens=cache_create,
            cache_read_tokens=cache_read,
        ),
    )
