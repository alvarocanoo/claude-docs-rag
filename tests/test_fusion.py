"""Reciprocal Rank Fusion unit tests — pure function, no I/O."""

from __future__ import annotations

from claude_docs_rag.retrieval.fusion import reciprocal_rank_fusion


def test_single_list_preserves_order() -> None:
    ranked = [("u1", 0), ("u2", 0), ("u3", 0)]
    fused = reciprocal_rank_fusion([ranked])
    keys = [(f.source_url, f.chunk_index) for f in fused]
    assert keys == ranked


def test_doc_in_both_lists_beats_doc_in_one() -> None:
    a = [("u1", 0), ("u2", 0), ("u3", 0)]
    b = [("u2", 0), ("u4", 0), ("u5", 0)]
    fused = reciprocal_rank_fusion([a, b])
    # u2 appears in both → must rank above u1 (in only one, even if first there)
    keys = [(f.source_url, f.chunk_index) for f in fused]
    assert keys[0] == ("u2", 0)


def test_higher_combined_rank_wins() -> None:
    a = [("u1", 0), ("u2", 0)]
    b = [("u1", 0), ("u3", 0)]
    fused = reciprocal_rank_fusion([a, b])
    assert (fused[0].source_url, fused[0].chunk_index) == ("u1", 0)
    assert fused[0].score > fused[1].score


def test_top_k_truncates() -> None:
    a = [(f"u{i}", 0) for i in range(10)]
    fused = reciprocal_rank_fusion([a], top_k=3)
    assert len(fused) == 3


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_const_affects_score_magnitude() -> None:
    a = [("u1", 0)]
    low_k = reciprocal_rank_fusion([a], k_const=1)[0].score
    high_k = reciprocal_rank_fusion([a], k_const=1000)[0].score
    assert low_k > high_k  # smaller k_const => higher contribution
