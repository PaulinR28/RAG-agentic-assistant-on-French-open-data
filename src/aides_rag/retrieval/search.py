"""The retrieval contract used by the agent: `search(query, k, filters) -> list[Hit]`.

One hit per aid (its best-scoring chunk), so `k` hits are `k` distinct aids, in rank order.
Metadata filters never drop aids that apply everywhere (national / European scope) or whose
region could not be resolved (`regions == []`, 47% of territorial aids, see DATA_CARD.md).

Ranking (`_scores`): hybrid BM25 + bge-m3, fused with weighted RRF (dense weight 3), the default measured in
eval/retrieval_fusion.json (held-out Recall@1 0.88, Recall@5 1.0). Each aid is represented by its best dense
chunk; aids outside both top-50 lists follow in dense order with a tiny score, so filters can still reach
every aid. The cross-encoder reranker is not used here (off by default: ~19 s per question on CPU).

Set AIDES_SEARCH=bm25 to fall back to BM25 only (no dense model in memory, e.g. on an 8 GB machine
that also runs the LLM).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

from aides_rag.index.chunks import read_corpus
from aides_rag.retrieval.bm25 import BM25
from aides_rag.retrieval.hybrid import DENSE_WEIGHT, rank_aids, rrf_scores

CHUNKS_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "chunks.jsonl"
EVERYWHERE = {"national", "european"}


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    id_aid: str
    score: float
    title: str
    text: str  # metadata header + body, what the LLM reads
    regions: list[str]
    date_fin: str | None
    aid_validation: str | None
    part: int = 1  # this chunk is part `part` of `n_parts` of the aid card
    n_parts: int = 1


@lru_cache(maxsize=1)
def _index() -> tuple[list[dict], BM25]:
    """Load the chunks and build the BM25 index once per process."""
    chunks = read_corpus(CHUNKS_PATH)
    return chunks, BM25([c["text"] for c in chunks])


@lru_cache(maxsize=1)
def _dense():
    """bge-m3 query encoder + the committed chunk vectors, loaded once per process (~2.6 GB RAM)."""
    from aides_rag.retrieval.dense import DEFAULT_DENSE, DenseIndex

    chunks, _ = _index()
    return DenseIndex(DEFAULT_DENSE, [c["text"] for c in chunks]), np.array([c["id_aid"] for c in chunks])


def _scores(query: str) -> np.ndarray:
    """One positive score per aid, on its representative chunk; 0 elsewhere. Higher = better."""
    chunks, bm25 = _index()
    b = bm25.scores(query)
    if os.getenv("AIDES_SEARCH", "hybrid") == "bm25":
        return b
    dense, chunk_aids = _dense()
    d = dense.scores(query)
    dense_ranking = rank_aids(d, chunk_aids)
    fused = rrf_scores([rank_aids(b, chunk_aids, positive_only=True), dense_ranking], weights=[1.0, DENSE_WEIGHT])
    tail = [aid for aid in dense_ranking if aid not in fused]
    score = {**fused, **{aid: 1e-6 / (r + 1) for r, aid in enumerate(tail)}}
    best_chunk: dict[str, int] = {}
    for i in np.argsort(-d, kind="stable"):
        best_chunk.setdefault(chunk_aids[i], int(i))
    out = np.zeros(len(chunks))
    for aid, value in score.items():
        out[best_chunk[aid]] = value
    return out


def _keep(chunk: dict, region: str | None, open_on: str | None, natures: list[str] | None) -> bool:
    if region and chunk["scope"] not in EVERYWHERE and chunk["regions"] and region not in chunk["regions"]:
        return False
    # ISO dates compare correctly as strings; an open-ended aid (date_fin None) is always open
    if open_on and chunk["date_fin"] and chunk["date_fin"] < open_on:
        return False
    if natures and not set(natures) & set(chunk["natures"]):
        return False
    return True


def search(query: str, k: int = 5, region: str | None = None,
           open_on: date | str | None = None, natures: list[str] | None = None) -> list[Hit]:
    """Top `k` aids for `query`, each represented by its best chunk."""
    chunks, _ = _index()
    open_on = open_on.isoformat() if isinstance(open_on, date) else open_on
    scores = _scores(query)
    hits: dict[str, Hit] = {}
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] <= 0 or len(hits) == k:
            break
        c = chunks[i]
        if c["id_aid"] in hits or not _keep(c, region, open_on, natures):
            continue
        hits[c["id_aid"]] = Hit(c["chunk_id"], c["id_aid"], float(scores[i]), c["title"], c["text"],
                                c["regions"], c["date_fin"], c["aid_validation"],
                                c["chunk_index"] + 1, c["n_chunks"])
    return list(hits.values())
