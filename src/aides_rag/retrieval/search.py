"""The retrieval contract used by the agent: `search(query, k, filters) -> list[Hit]`.

One hit per aid (its best-scoring chunk), so `k` hits are `k` distinct aids, in rank order.
Metadata filters never drop aids that apply everywhere (national / European scope) or whose
region could not be resolved (`regions == []`, 47% of territorial aids, see DATA_CARD.md).

Current implementation: BM25 only (stub). Workstream A replaces `_scores` with hybrid
BM25 + dense + reranking; the signature and the Hit fields stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

from aides_rag.index.chunks import read_corpus
from aides_rag.retrieval.bm25 import BM25

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


@lru_cache(maxsize=1)
def _index() -> tuple[list[dict], BM25]:
    """Load the chunks and build the BM25 index once per process."""
    chunks = read_corpus(CHUNKS_PATH)
    return chunks, BM25([c["text"] for c in chunks])


def _scores(query: str) -> np.ndarray:
    chunks, bm25 = _index()
    return bm25.scores(query)


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
                                c["regions"], c["date_fin"], c["aid_validation"])
    return list(hits.values())
