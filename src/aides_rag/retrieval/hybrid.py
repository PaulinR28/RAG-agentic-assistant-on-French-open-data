"""Hybrid retrieval: BM25 + dense rankings merged with Reciprocal Rank Fusion (RRF).

Both retrievers score every chunk; chunk scores are turned into an aid ranking (best chunk per aid),
then the two aid rankings are fused: score(aid) = sum over retrievers of 1 / (K + rank).
RRF only uses ranks, so the very different score scales of BM25 and cosine never need calibrating.
K = 60 is the usual constant (Cormack et al., 2009); only the top DEPTH aids of each list count.
"""

from __future__ import annotations

import numpy as np

RRF_K = 60
DEPTH = 50


def rank_aids(scores: np.ndarray, chunk_aids: np.ndarray, positive_only: bool = False) -> list[str]:
    """Aids ordered by their best chunk score. BM25 passes positive_only (score 0 = no shared term)."""
    ranking: dict[str, None] = {}
    for i in np.argsort(-scores, kind="stable"):
        if positive_only and scores[i] <= 0:
            break
        ranking.setdefault(chunk_aids[i], None)
    return list(ranking)


def rrf(rankings: list[list], k: int = RRF_K, depth: int = DEPTH, weights: list[float] | None = None) -> list:
    """Fuse rankings of aid ids (or chunk indices); ties broken by id, so the result is deterministic.

    weights (one per ranking, default all 1) give one retriever more say: score += w / (k + rank).
    """
    fused: dict = {}
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings)):
        for rank, item in enumerate(ranking[:depth], start=1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
    return sorted(fused, key=lambda item: (-fused[item], item))


def top_chunks(scores: np.ndarray, depth: int = DEPTH, positive_only: bool = False) -> list[int]:
    order = np.argsort(-scores, kind="stable")[:depth]
    return [int(i) for i in order if not positive_only or scores[i] > 0]


def hybrid_chunks(bm25_scores: np.ndarray, dense_scores: np.ndarray, depth: int = DEPTH) -> list[int]:
    """Chunk indices ordered by RRF of the BM25 and dense chunk rankings (the reranker's candidate pool)."""
    return rrf([top_chunks(bm25_scores, depth, positive_only=True), top_chunks(dense_scores, depth)], depth=depth)


def aids_in_order(chunk_order: list[int], chunk_aids: np.ndarray) -> list[str]:
    return list(dict.fromkeys(chunk_aids[i] for i in chunk_order))
