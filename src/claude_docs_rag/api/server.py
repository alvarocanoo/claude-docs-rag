"""FastAPI HTTP surface for claude-docs-rag.

Endpoints:
  GET  /healthz            liveness + DB and BM25 index status
  GET  /metrics            basic counters (requests, latencies, errors)
  POST /search             hybrid retrieval -> JSON (no LLM call, no API key needed)
  POST /ask                end-to-end RAG -> JSON answer with citations + cost
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from claude_docs_rag import __version__
from claude_docs_rag.agent.pipeline import answer_question
from claude_docs_rag.retrieval.hybrid import hybrid_search
from claude_docs_rag.retrieval.sparse import INDEX_DIR as BM25_INDEX_DIR
from claude_docs_rag.settings import settings
from claude_docs_rag.storage.vector_store import count_documents

# Windows fixups need to land before uvicorn starts the loop, when serving
# from `cdrag serve`. Importing this module from tests is fine either way.
if sys.platform == "win32":
    with suppress(RuntimeError):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# --------- Pydantic schemas ---------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    documents_count: int
    bm25_index_present: bool


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    source_url: str
    title: str
    section_path: str
    rerank_score: float
    fusion_score: float
    excerpt: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    model: str | None = None
    max_tokens: int = Field(default=1024, ge=64, le=4096)


class CitationOut(BaseModel):
    chunk_id: int
    source_url: str
    section_path: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class MetricsResponse(BaseModel):
    requests_total: int
    requests_by_endpoint: dict[str, int]
    last_50_latencies_ms: dict[str, list[float]]
    errors_total: int


# --------- App state ---------------------------------------------------------


def _init_state(app: FastAPI) -> None:
    app.state.started_at = time.time()
    app.state.requests_total = 0
    app.state.requests_by_endpoint = {"search": 0, "ask": 0, "healthz": 0, "metrics": 0}
    app.state.latencies = {"search": deque(maxlen=50), "ask": deque(maxlen=50)}
    app.state.errors_total = 0


async def _warmup() -> None:
    """Run a full hybrid_search once so embedder, BM25 index, reranker and the
    DB connection are all primed before the first user request."""
    from claude_docs_rag.retrieval.hybrid import hybrid_search

    await hybrid_search("warmup", top_k_rerank=1)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Warmup is best-effort; if a model is missing the server still boots and
    # individual requests will surface a clear error.
    with suppress(Exception):
        await _warmup()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="claude-docs-rag",
        version=__version__,
        description="Hybrid RAG over Anthropic Claude API docs.",
        lifespan=_lifespan,
    )
    # Permissive CORS for the local dev frontend. In production this should be
    # tightened to the actual deploy origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    _init_state(app)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        app.state.requests_total += 1
        app.state.requests_by_endpoint["healthz"] += 1
        try:
            n = await count_documents()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"db unreachable: {exc}") from exc
        return HealthResponse(
            status="ok",
            version=__version__,
            documents_count=n,
            bm25_index_present=Path(BM25_INDEX_DIR).exists(),
        )

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        app.state.requests_total += 1
        app.state.requests_by_endpoint["metrics"] += 1
        return MetricsResponse(
            requests_total=app.state.requests_total,
            requests_by_endpoint=dict(app.state.requests_by_endpoint),
            last_50_latencies_ms={k: list(v) for k, v in app.state.latencies.items()},
            errors_total=app.state.errors_total,
        )

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        app.state.requests_total += 1
        app.state.requests_by_endpoint["search"] += 1
        started = time.perf_counter()
        try:
            results = await hybrid_search(req.query, top_k_rerank=req.k)
        except FileNotFoundError as exc:
            app.state.errors_total += 1
            raise HTTPException(
                status_code=503,
                detail="BM25 index missing — run `cdrag build-bm25`.",
            ) from exc
        except Exception as exc:
            app.state.errors_total += 1
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        app.state.latencies["search"].append(elapsed_ms)

        return SearchResponse(
            query=req.query,
            hits=[
                SearchHit(
                    source_url=r.source_url,
                    title=r.title,
                    section_path=r.section_path,
                    rerank_score=r.rerank_score,
                    fusion_score=r.fusion_score,
                    excerpt=" ".join(r.content.split())[:240],
                )
                for r in results
            ],
            latency_ms=elapsed_ms,
        )

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        if not settings.anthropic_api_key:
            raise HTTPException(
                status_code=503,
                detail="ANTHROPIC_API_KEY not configured.",
            )
        app.state.requests_total += 1
        app.state.requests_by_endpoint["ask"] += 1
        started = time.perf_counter()
        try:
            result = await answer_question(
                req.question,
                model=req.model,
                top_k_rerank=req.k,
                max_tokens=req.max_tokens,
            )
        except Exception as exc:
            app.state.errors_total += 1
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        app.state.latencies["ask"].append(elapsed_ms)

        call = result.call
        return AskResponse(
            question=result.question,
            answer=result.answer,
            citations=[
                CitationOut(
                    chunk_id=c.chunk_id,
                    source_url=c.source_url,
                    section_path=c.section_path,
                )
                for c in result.citations
            ],
            model=call.model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            cache_creation_tokens=call.cache_creation_tokens,
            cache_read_tokens=call.cache_read_tokens,
            cost_usd=call.cost_usd,
            retrieval_ms=result.timings.get("retrieval", 0) * 1000,
            generation_ms=result.timings.get("generation", 0) * 1000,
            total_ms=elapsed_ms,
        )

    return app


app = create_app()
