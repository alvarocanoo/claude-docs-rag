"""Sparse retrieval over chunk content using bm25s.

A BM25 index is built once after ingest (see scripts/build_bm25.py) and dumped
to disk. At query time we load the index in memory; the index is small (a few MB
for ~75 k chunks) so this is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import bm25s
from pydantic import BaseModel

INDEX_DIR = Path("data/bm25_index")
META_FILE = INDEX_DIR / "meta.jsonl"


class SparseHit(BaseModel):
    source_url: str
    chunk_index: int
    score: float


class _Meta(TypedDict):
    source_url: str
    chunk_index: int


class _Bundle:
    """Lazily-loaded BM25 retriever + parallel metadata array."""

    _retriever: bm25s.BM25 | None = None
    _meta: list[_Meta] | None = None

    @classmethod
    def load(cls) -> tuple[bm25s.BM25, list[_Meta]]:
        if cls._retriever is None or cls._meta is None:
            if not INDEX_DIR.exists():
                raise FileNotFoundError(
                    f"BM25 index not found at {INDEX_DIR}. Run `cdrag build-bm25` first."
                )
            cls._retriever = bm25s.BM25.load(str(INDEX_DIR), load_corpus=False)
            cls._meta = [
                json.loads(line)
                for line in META_FILE.read_text(encoding="utf-8").splitlines()
                if line
            ]
        return cls._retriever, cls._meta


def build_index(documents: list[tuple[str, int, str]]) -> None:
    """Build and persist a BM25 index from (source_url, chunk_index, content) tuples."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    texts = [d[2] for d in documents]
    tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(str(INDEX_DIR), corpus=None)

    META_FILE.write_text(
        "\n".join(json.dumps({"source_url": d[0], "chunk_index": d[1]}) for d in documents),
        encoding="utf-8",
    )
    _Bundle._retriever = None  # invalidate cache
    _Bundle._meta = None


def search(query: str, *, k: int = 20) -> list[SparseHit]:
    retriever, meta = _Bundle.load()
    query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    hits: list[SparseHit] = []
    for idx, score in zip(results[0], scores[0], strict=True):
        m = meta[int(idx)]
        hits.append(
            SparseHit(
                source_url=m["source_url"],
                chunk_index=m["chunk_index"],
                score=float(score),
            )
        )
    return hits
