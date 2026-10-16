"""Retrieval system: BM25 + dense (bge-m3 by default) fused with RRF, no LLM. It only fills `retrieved_ids`.

Same fusion as retrieval/bench_dense.py ("hybrid" rows of eval/retrieval_dense.json), so its
Recall@k must match those numbers. Needs the dense model locally (models/<alias>/ or the Hugging Face
cache); chunk vectors are read from indexes/ when present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aides_rag.index.chunks import read_corpus
from aides_rag.retrieval.bm25 import BM25
from aides_rag.retrieval.dense import DEFAULT_DENSE, DenseIndex
from aides_rag.retrieval.hybrid import rank_aids, rrf


class Hybrid:
    def __init__(self, chunks_path: Path = Path("data/processed/chunks.jsonl"), dense: str = DEFAULT_DENSE,
                 k: int = 10):
        chunks = read_corpus(chunks_path)
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25(texts, stem=True)
        self.dense = DenseIndex(dense, texts)
        self.chunk_aids = np.array([c["id_aid"] for c in chunks])
        self.k = k

    def answer(self, question: str) -> dict:
        ranking = rrf([rank_aids(self.bm25.scores(question), self.chunk_aids, positive_only=True),
                       rank_aids(self.dense.scores(question), self.chunk_aids)])
        return {
            "answer": None,
            "cited_sources": [],
            "retrieved_ids": ranking[: self.k],
            "tokens_in": None,
            "tokens_out": None,
            "tool_calls": [],
            "stop_reason": None,
        }
