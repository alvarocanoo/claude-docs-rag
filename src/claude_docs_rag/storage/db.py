"""Async Postgres connection helper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from pgvector.psycopg import register_vector_async

from claude_docs_rag.settings import settings


@asynccontextmanager
async def connect() -> AsyncIterator[psycopg.AsyncConnection]:
    """Async connection with pgvector type adapters registered."""
    async with await psycopg.AsyncConnection.connect(settings.postgres_dsn) as conn:
        await register_vector_async(conn)
        yield conn
