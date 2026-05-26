"""Unit tests for the MCP server module — verify the tools are registered
with the expected names and signatures, without spawning the stdio loop."""

from __future__ import annotations

import asyncio

import pytest

from claude_docs_rag.mcp.server import mcp


def test_tools_registered() -> None:
    """`search_claude_docs` and `ask_claude_docs` must both be discoverable."""
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "search_claude_docs" in names
    assert "ask_claude_docs" in names


def test_search_tool_schema_shape() -> None:
    tool = next(t for t in asyncio.run(mcp.list_tools()) if t.name == "search_claude_docs")
    # FastMCP derives the JSON schema from the type hints; the only fields we
    # commit to are: tool has a description (so MCP clients can show it) and
    # it accepts the `query` and `k` parameters we declared.
    assert tool.description
    schema = tool.inputSchema
    assert "properties" in schema
    assert "query" in schema["properties"]
    assert "k" in schema["properties"]


def test_ask_tool_schema_shape() -> None:
    tool = next(t for t in asyncio.run(mcp.list_tools()) if t.name == "ask_claude_docs")
    assert tool.description
    schema = tool.inputSchema
    assert "question" in schema["properties"]
    assert "k" in schema["properties"]
    assert "max_tokens" in schema["properties"]


def test_ask_tool_refuses_without_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """When neither ANTHROPIC_API_KEY nor GROQ_API_KEY is configured, the
    `ask_claude_docs` tool must surface a clear error instead of silently
    calling the LLM with empty credentials. FastMCP wraps tool exceptions
    into `ToolError` before propagating them to the MCP client."""
    from mcp.server.fastmcp.exceptions import ToolError

    from claude_docs_rag.settings import settings

    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "")

    async def _invoke() -> None:
        await mcp.call_tool("ask_claude_docs", {"question": "hi", "k": 3, "max_tokens": 64})

    with pytest.raises(ToolError, match="GROQ_API_KEY"):
        asyncio.run(_invoke())
