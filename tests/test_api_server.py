"""Integration tests for the FastAPI server. Skipped if DB / index unavailable."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from claude_docs_rag.api.server import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz_responds(client: TestClient) -> None:
    """Health endpoint should answer 200 or 503 (db unreachable from CI)."""
    resp = client.get("/healthz")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert body["status"] == "ok"
        assert body["documents_count"] >= 0
        assert isinstance(body["bm25_index_present"], bool)


def test_metrics_initial_state(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["requests_total"] >= 1  # this very request counted
    assert "search" in body["requests_by_endpoint"]
    assert "ask" in body["requests_by_endpoint"]
    assert body["errors_total"] == 0


def test_search_validation_rejects_empty_query(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "", "k": 5})
    assert resp.status_code == 422  # pydantic validation


def test_search_validation_rejects_huge_k(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "test", "k": 999})
    assert resp.status_code == 422


def test_ask_503_when_anthropic_provider_has_no_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With LLM_PROVIDER=anthropic and no ANTHROPIC_API_KEY, /ask refuses 503."""
    from claude_docs_rag.api import server as server_mod

    monkeypatch.setattr(server_mod.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(server_mod.settings, "anthropic_api_key", "")
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 503
    body = resp.json()
    assert "ANTHROPIC_API_KEY" in body["detail"]
    assert "'anthropic'" in body["detail"]


def test_ask_503_when_groq_provider_has_no_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With LLM_PROVIDER=groq and no GROQ_API_KEY, /ask refuses 503."""
    from claude_docs_rag.api import server as server_mod

    monkeypatch.setattr(server_mod.settings, "llm_provider", "groq")
    monkeypatch.setattr(server_mod.settings, "groq_api_key", "")
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 503
    body = resp.json()
    assert "GROQ_API_KEY" in body["detail"]
    assert "'groq'" in body["detail"]
