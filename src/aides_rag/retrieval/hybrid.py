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


def rrf(rankings: list[list[str]], k: int = RRF_K, depth: int = DEPTH) -> list[str]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, aid in enumerate(ranking[:depth], start=1):
            fused[aid] = fused.get(aid, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda aid: (-fused[aid], aid))
