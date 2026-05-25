"""Persistence + retrieval over the `documents` table (pgvector + FTS)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from claude_docs_rag.models import Chunk
from claude_docs_rag.storage.db import connect
from claude_docs_rag.storage.schema import SCHEMA_SQL


class RetrievedChunk(BaseModel):
    source_url: str
    title: str
    section_path: str
    chunk_index: int
    content: str
    score: float


async def apply_schema() -> None:
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(SCHEMA_SQL)
        await conn.commit()


async def count_documents() -> int:
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SELECT COUNT(*) FROM documents")
        row = await cur.fetchone()
        return int(row[0]) if row else 0


async def upsert_chunks(chunks: list[Chunk]) -> int:
    """Insert / update chunks. Returns the number of rows written."""
    if not chunks:
        return 0

    rows = [
        (
            c.source_url,
            c.title,
            c.section_path,
            c.chunk_index,
            c.content,
            c.content_tokens,
            np.asarray(c.embedding, dtype=np.float32) if c.embedding else None,
        )
        for c in chunks
    ]

    sql = """
        INSERT INTO documents
            (source_url, title, section_path, chunk_index, content, content_tokens, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_url, chunk_index) DO UPDATE SET
            title          = EXCLUDED.title,
            section_path   = EXCLUDED.section_path,
            content        = EXCLUDED.content,
            content_tokens = EXCLUDED.content_tokens,
            embedding      = EXCLUDED.embedding,
            ingested_at    = NOW()
    """
    async with connect() as conn, conn.cursor() as cur:
        await cur.executemany(sql, rows)
        await conn.commit()
        return cur.rowcount or 0


async def search_semantic(
    query_embedding: list[float],
    *,
    k: int = 20,
) -> list[RetrievedChunk]:
    """Cosine-similarity top-K via pgvector HNSW."""
    vec = np.asarray(query_embedding, dtype=np.float32)
    sql = """
        SELECT source_url, title, section_path, chunk_index, content,
               1 - (embedding <=> %s) AS score
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(sql, (vec, vec, k))
        rows = await cur.fetchall()

    return [
        RetrievedChunk(
            source_url=r[0],
            title=r[1],
            section_path=r[2],
            chunk_index=r[3],
            content=r[4],
            score=float(r[5]),
        )
        for r in rows
    ]
