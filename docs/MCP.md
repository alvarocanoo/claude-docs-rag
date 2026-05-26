# MCP server — claude-docs-rag tools for any Claude client

The repo ships a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes the RAG as **tools** any MCP-compatible client can call. Hook it into Claude Desktop, the Claude CLI, or any agent built on the official MCP SDKs, and the agent gains:

| Tool                  | Purpose                                                                                                                       | Needs LLM key? |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------|----------------|
| `search_claude_docs`  | Hybrid retrieval over 42 k chunks of `platform.claude.com/docs`. Returns top-K reranked chunks with section, URL, excerpt.    | No             |
| `ask_claude_docs`     | Full RAG. Retrieves and asks the configured LLM (Groq Llama 3.1 8B default) to answer with inline `[n]` citations + cost.    | Yes            |

The server runs on stdio (the transport Claude Desktop expects) via `cdrag mcp`.

## Why this exists

The Anthropic docs are large enough that pasting them in-context per question is wasteful. With the MCP server wired in, *your* Claude session can ground itself on the official docs every time the conversation touches the API — no copy-paste, automatic citations, ~3-4 s per call.

It's also a clean demo of [ADR-001…ADR-012](DECISIONS.md) acting as a callable surface, not just an HTTP API.

## Setup — Claude Desktop

Edit `claude_desktop_config.json` (on macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "claude-docs-rag": {
      "command": "uv",
      "args": ["--directory", "C:/Users/acano/dev/claude-docs-rag", "run", "cdrag", "mcp"],
      "env": {
        "POSTGRES_DSN": "postgresql://...neon.tech/neondb?sslmode=require",
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "gsk_...",
        "GROQ_MODEL": "llama-3.1-8b-instant"
      }
    }
  }
}
```

Restart Claude Desktop. The tools appear in the **🛠 Tools** panel of the chat sidebar. The first call cold-starts the embedder + reranker (~10 s); subsequent calls run in the 3-4 s range observed locally.

If you don't want `ask_claude_docs` enabled (e.g. you'd rather have the host model answer, not the 8 B), simply leave `GROQ_API_KEY` unset — `search_claude_docs` keeps working and the host LLM consumes the chunks directly.

## Setup — Claude CLI / any MCP client

```bash
# the server is just stdio; any MCP client that can spawn a subprocess works
uv --directory /path/to/claude-docs-rag run cdrag mcp
```

For programmatic clients:

```python
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

server = StdioServerParameters(
    command="uv",
    args=["--directory", "/path/to/claude-docs-rag", "run", "cdrag", "mcp"],
    env={"POSTGRES_DSN": "...", "LLM_PROVIDER": "groq", "GROQ_API_KEY": "..."},
)

async with stdio_client(server) as (read, write):
    # use the read/write streams or the higher-level mcp.client APIs
    ...
```

## Tool signatures

### `search_claude_docs(query: str, k: int = 5) -> list[dict]`

Hybrid retrieval (BM25 + dense + cross-encoder rerank). Each result:

```json
{
  "source_url": "https://platform.claude.com/docs/en/build-with-claude/streaming.md",
  "title": "Streaming messages",
  "section_path": "Streaming messages > Full HTTP stream response > Basic streaming request",
  "rerank_score": 6.932,
  "fusion_score": 0.015,
  "excerpt": "Streaming messages > Full HTTP stream response > Basic streaming request <CodeGroup> ```bash cURL curl https://api.anthropic.com/v1/messages \\ --header..."
}
```

### `ask_claude_docs(question: str, k: int = 5, max_tokens: int = 800) -> dict`

End-to-end RAG. The answer includes inline `[n]` markers pointing at the `citations` array.

```json
{
  "question": "How do I stream messages from the Claude API in Python?",
  "answer": "You can stream messages using the `anthropic` library [1]. ...",
  "citations": [
    {"chunk_id": 1, "source_url": "https://...streaming.md", "section_path": "..."}
  ],
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "input_tokens": 3311,
  "output_tokens": 249,
  "cost_usd": 0.00018,
  "timings_ms": {"retrieval": 2800, "generation": 900, "total": 3700}
}
```

## Notes

- The MCP server uses the **same retrieval pipeline** as `/search` and `/ask/stream`. Any improvement to retrieval (HyDE ADR-011, sparse-recovery ADR-012, future ADR-014) lights up here for free.
- Cold start is ~10 s the first time per session (model loading); after that, requests are sub-second on the retrieval side.
- Cost accounting is identical to the HTTP path — the `cost_usd` field is what the call *would* cost at paid-tier prices, even when the free-tier consumed it for $0.
