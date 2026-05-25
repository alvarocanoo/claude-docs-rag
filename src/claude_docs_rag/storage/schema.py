"""Database schema DDL — runnable as an idempotent migration."""

from __future__ import annotations

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    source_url      TEXT NOT NULL,
    title           TEXT NOT NULL,
    section_path    TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    content_tokens  INTEGER NOT NULL,
    embedding       vector(384),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_url, chunk_index)
);

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS documents_content_fts_idx
    ON documents USING gin (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS documents_source_url_idx
    ON documents (source_url);
"""
