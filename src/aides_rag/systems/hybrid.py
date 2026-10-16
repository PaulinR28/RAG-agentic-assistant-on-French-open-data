"""Retrieval systems: BM25 + dense (bge-m3) fused with weighted RRF, no LLM. They only fill `retrieved_ids`.

- `Hybrid` (default, run_eval --system hybrid): aid-level RRF with dense weighted DENSE_WEIGHT (3) against
  BM25 (1). Its Recall@k must match the "rrf_w3" rows of eval/retrieval_fusion.json.
- `HybridRerank` (option, --system hybrid_rerank): the top RERANK_POOL chunks of the same fusion re-sorted by
  the bge-reranker-v2-m3 cross-encoder. Off by default: on this laptop's CPU it adds ~19 s per question for
  no held-out gain (eval/retrieval_rerank.json).

Needs the models locally (models/<alias>/ or the Hugging Face cache); chunk vectors are read from indexes/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aides_rag.index.chunks import read_corpus
from aides_rag.retrieval.bm25 import BM25
from aides_rag.retrieval.dense import DEFAULT_DENSE, DenseIndex
from aides_rag.retrieval.hybrid import DENSE_WEIGHT, aids_in_order, hybrid_chunks, rank_aids, rrf

RERANK_POOL = 10


def _empty_answer(retrieved: list[str]) -> dict:
    return {"answer": None, "cited_sources": [], "retrieved_ids": retrieved, "tokens_in": None, "tokens_out": None,
            "tool_calls": [], "stop_reason": None}


class Hybrid:
    def __init__(self, chunks_path: Path = Path("data/processed/chunks.jsonl"), dense: str = DEFAULT_DENSE,
                 dense_weight: float = DENSE_WEIGHT, k: int = 10):
        chunks = read_corpus(chunks_path)
        self.texts = [c["text"] for c in chunks]
        self.bm25 = BM25(self.texts, stem=True)
        self.dense = DenseIndex(dense, self.texts)
        self.chunk_aids = np.array([c["id_aid"] for c in chunks])
        self.dense_weight = dense_weight
        self.k = k

    def answer(self, question: str) -> dict:
        ranking = rrf([rank_aids(self.bm25.scores(question), self.chunk_aids, positive_only=True),
                       rank_aids(self.dense.scores(question), self.chunk_aids)], weights=[1.0, self.dense_weight])
        return _empty_answer(ranking[: self.k])


class HybridRerank(Hybrid):
    def __init__(self, reranker: str = "bge-reranker", pool: int = RERANK_POOL, **kwargs):
        from aides_rag.retrieval.reranker import Reranker

        super().__init__(**kwargs)
        self.reranker = Reranker(reranker)
        self.pool = pool

    def answer(self, question: str) -> dict:
        fused = hybrid_chunks(self.bm25.scores(question), self.dense.scores(question), dense_weight=self.dense_weight)
        reranked, _ = self.reranker.rerank(question, fused[: self.pool], self.texts, batch_size=4)
        return _empty_answer(aids_in_order(reranked + fused[self.pool:], self.chunk_aids)[: self.k])
