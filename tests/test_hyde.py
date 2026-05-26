"""Unit tests for HyDE query expansion (ADR-011). No network calls."""

from __future__ import annotations

import asyncio

import pytest

from claude_docs_rag.agent.client import CallResult
from claude_docs_rag.retrieval import hyde


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.get_event_loop().run_until_complete(coro)


def _fake_call_result(text: str) -> CallResult:
    return CallResult(
        text=text,
        provider="test",
        model="test-model",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        stop_reason="stop",
        cost_usd=0.0,
    )


def test_hyde_disabled_returns_raw_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hyde.settings, "hyde_enabled", False)
    out = _run(hyde.expand_for_dense("how do I stream messages?"))
    assert out == "how do I stream messages?"


def test_hyde_no_llm_configured_returns_raw_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hyde.settings, "hyde_enabled", True)
    monkeypatch.setattr(hyde.settings, "llm_provider", "groq")
    monkeypatch.setattr(hyde.settings, "groq_api_key", "")
    monkeypatch.setattr(hyde.settings, "anthropic_api_key", "")
    out = _run(hyde.expand_for_dense("test query"))
    assert out == "test query"


def test_hyde_appends_passage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hyde.settings, "hyde_enabled", True)
    monkeypatch.setattr(hyde.settings, "llm_provider", "groq")
    monkeypatch.setattr(hyde.settings, "groq_api_key", "gsk_fake_for_test")

    async def fake_create_message(**kwargs):  # type: ignore[no-untyped-def]
        return _fake_call_result(
            "Streaming uses messages.stream() which returns an async iterator..."
        )

    monkeypatch.setattr(hyde, "create_message", fake_create_message)
    out = _run(hyde.expand_for_dense("how do I stream?"))
    assert out.startswith("how do I stream?")
    assert "messages.stream" in out
    assert "\n\n" in out  # query and passage are separated


def test_hyde_empty_passage_falls_back_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hyde.settings, "hyde_enabled", True)
    monkeypatch.setattr(hyde.settings, "llm_provider", "groq")
    monkeypatch.setattr(hyde.settings, "groq_api_key", "gsk_fake_for_test")

    async def fake_create_message(**kwargs):  # type: ignore[no-untyped-def]
        return _fake_call_result("   ")  # whitespace-only

    monkeypatch.setattr(hyde, "create_message", fake_create_message)
    out = _run(hyde.expand_for_dense("test"))
    assert out == "test"


def test_hyde_exception_falls_back_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hyde.settings, "hyde_enabled", True)
    monkeypatch.setattr(hyde.settings, "llm_provider", "groq")
    monkeypatch.setattr(hyde.settings, "groq_api_key", "gsk_fake_for_test")

    async def fake_create_message(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(hyde, "create_message", fake_create_message)
    out = _run(hyde.expand_for_dense("test"))
    assert out == "test"


def test_hyde_timeout_falls_back_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hyde.settings, "hyde_enabled", True)
    monkeypatch.setattr(hyde.settings, "llm_provider", "groq")
    monkeypatch.setattr(hyde.settings, "groq_api_key", "gsk_fake_for_test")
    monkeypatch.setattr(hyde.settings, "hyde_timeout_seconds", 0.05)

    async def slow_create_message(**kwargs):  # type: ignore[no-untyped-def]
        await asyncio.sleep(2.0)
        return _fake_call_result("never reached")

    monkeypatch.setattr(hyde, "create_message", slow_create_message)
    out = _run(hyde.expand_for_dense("test"))
    assert out == "test"
