"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3, local)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

from sentence_transformers import CrossEncoder

from claude_docs_rag.settings import settings


@lru_cache(maxsize=1)
def _load_reranker(name: str) -> CrossEncoder:
    return cast(CrossEncoder, CrossEncoder(name, trust_remote_code=False))


def rerank(
    query: str,
    passages: list[str],
    *,
    model_name: str | None = None,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Return (original_index, score) ordered by relevance, high to low.

    Empty `passages` returns an empty list.
    """
    if not passages:
        return []
    model = _load_reranker(model_name or settings.reranker_model)
    pairs: Any = [(query, p) for p in passages]
    scores: Any = model.predict(pairs, show_progress_bar=False)
    indexed = list(enumerate(float(s) for s in scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k] if top_k is not None else indexed
