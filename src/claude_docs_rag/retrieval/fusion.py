"""Reciprocal Rank Fusion — combines ranked lists from heterogeneous retrievers.

RRF is rank-based (not score-based), so it normalises the very different scales
of cosine similarity (0..1) and BM25 (unbounded > 0). See Cormack et al. 2009.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_K_CONST = 60


@dataclass(frozen=True)
class FusionItem:
    source_url: str
    chunk_index: int
    score: float

    @property
    def key(self) -> tuple[str, int]:
        return (self.source_url, self.chunk_index)


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[tuple[str, int]]],
    *,
    k_const: int = DEFAULT_K_CONST,
    top_k: int | None = None,
) -> list[FusionItem]:
    """Each input is an ordered list of (source_url, chunk_index) — best first.

    Returns FusionItems sorted by fused score (high to low).
    """
    accum: dict[tuple[str, int], float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            accum[key] = accum.get(key, 0.0) + 1.0 / (k_const + rank)

    items = [FusionItem(k[0], k[1], v) for k, v in accum.items()]
    items.sort(key=lambda x: x.score, reverse=True)
    return items[:top_k] if top_k is not None else items
