"""Baseline system: French BM25 over the chunks, no LLM. It only fills `retrieved_ids`.

Same index as the default variant of retrieval/eval_recall.py (stemming + metadata header),
so its Recall@k must match `eval/retrieval_bm25.json`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aides_rag.index.chunks import read_corpus
from aides_rag.retrieval.bm25 import BM25
from aides_rag.retrieval.eval_recall import rank_aids


class Bm25Only:
    def __init__(self, chunks_path: Path = Path("data/processed/chunks.jsonl"), k: int = 10):
        chunks = read_corpus(chunks_path)
        self.index = BM25([c["text"] for c in chunks], stem=True)
        self.chunk_aids = np.array([c["id_aid"] for c in chunks])
        self.k = k

    def answer(self, question: str) -> dict:
        return {
            "answer": None,
            "cited_sources": [],
            "retrieved_ids": rank_aids(self.index, self.chunk_aids, question)[: self.k],
            "tokens_in": None,
            "tokens_out": None,
            "tool_calls": [],
            "stop_reason": None,
        }
