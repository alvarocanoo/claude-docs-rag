"""Local embedding using BAAI/bge-m3 via sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

from sentence_transformers import SentenceTransformer

from claude_docs_rag.models import Chunk
from claude_docs_rag.settings import settings

EMBEDDING_DIM = 1024


@lru_cache(maxsize=2)
def _load_model(name: str) -> SentenceTransformer:
    return cast(SentenceTransformer, SentenceTransformer(name, trust_remote_code=False))


def embed_texts(
    texts: list[str],
    *,
    model_name: str | None = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> list[list[float]]:
    """Return one embedding vector per input text. Normalized for cosine similarity."""
    model = _load_model(model_name or settings.embedding_model)
    arr: Any = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if arr.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension {arr.shape[1]} != expected {EMBEDDING_DIM} for {model_name}"
        )
    return cast(list[list[float]], arr.tolist())


def embed_chunks(chunks: list[Chunk], **kwargs: Any) -> list[Chunk]:
    """Mutate chunks in place — attach .embedding to each."""
    if not chunks:
        return chunks
    vectors = embed_texts([c.content for c in chunks], **kwargs)
    for chunk, vec in zip(chunks, vectors, strict=True):
        chunk.embedding = vec
    return chunks
