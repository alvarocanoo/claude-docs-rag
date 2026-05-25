"""Async Postgres connection helper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from pgvector.psycopg import register_vector_async

from claude_docs_rag.settings import settings


@asynccontextmanager
async def connect(*, register_vector: bool = True) -> AsyncIterator[psycopg.AsyncConnection]:
    """Async connection. By default registers pgvector adapters — disable for
    schema-bootstrap connections that run CREATE EXTENSION vector themselves."""
    async with await psycopg.AsyncConnection.connect(settings.postgres_dsn) as conn:
        if register_vector:
            await register_vector_async(conn)
        yield conn
