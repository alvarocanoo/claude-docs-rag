"""Unit tests for the provider-pluggable LLM client (no network calls)."""

from __future__ import annotations

import pytest

from claude_docs_rag.agent.client import (
    _cost_anthropic,
    _cost_groq,
    _flatten_system_to_text,
    _resolve_model,
    _resolve_provider,
)

# ---------- _resolve_provider ----------


def test_resolve_provider_lowercases() -> None:
    assert _resolve_provider("ANTHROPIC") == "anthropic"
    assert _resolve_provider("Groq") == "groq"


def test_resolve_provider_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        _resolve_provider("openai")


def test_resolve_provider_falls_back_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from claude_docs_rag.agent import client as client_mod

    monkeypatch.setattr(client_mod.settings, "llm_provider", "groq")
    assert _resolve_provider(None) == "groq"


# ---------- _resolve_model ----------


def test_resolve_model_uses_explicit_override() -> None:
    assert _resolve_model("anthropic", "claude-opus-4-7") == "claude-opus-4-7"
    assert _resolve_model("groq", "llama-3.1-8b-instant") == "llama-3.1-8b-instant"


def test_resolve_model_default_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from claude_docs_rag.agent import client as client_mod

    monkeypatch.setattr(client_mod.settings, "model_simple", "claude-haiku-4-5-20251001")
    monkeypatch.setattr(client_mod.settings, "groq_model", "llama-3.3-70b-versatile")
    assert _resolve_model("anthropic", None) == "claude-haiku-4-5-20251001"
    assert _resolve_model("groq", None) == "llama-3.3-70b-versatile"


# ---------- _flatten_system_to_text ----------


def test_flatten_string_passthrough() -> None:
    assert _flatten_system_to_text("hello") == "hello"


def test_flatten_text_blocks_joined_with_blank_line() -> None:
    blocks = [
        {"type": "text", "text": "system rules"},
        {"type": "text", "text": "CONTEXT:\n[1] foo", "cache_control": {"type": "ephemeral"}},
    ]
    out = _flatten_system_to_text(blocks)  # type: ignore[arg-type]
    assert "system rules" in out
    assert "CONTEXT:" in out
    assert "\n\n" in out


def test_flatten_ignores_non_text_blocks() -> None:
    blocks = [{"type": "image", "source": {"data": "..."}}]
    assert _flatten_system_to_text(blocks) == ""  # type: ignore[arg-type]


# ---------- _cost_* ----------


def test_cost_anthropic_haiku_known_model() -> None:
    cost = _cost_anthropic(
        model="claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )
    assert cost == pytest.approx(1.00)


def test_cost_anthropic_unknown_model_returns_zero() -> None:
    assert (
        _cost_anthropic(
            model="claude-future-99",
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
        == 0.0
    )


def test_cost_groq_llama_known_model() -> None:
    cost = _cost_groq(
        model="llama-3.3-70b-versatile", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == pytest.approx(0.59 + 0.79)


def test_cost_groq_unknown_model_returns_zero() -> None:
    assert _cost_groq(model="qwen-future", input_tokens=1000, output_tokens=500) == 0.0
