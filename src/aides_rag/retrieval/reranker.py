"""Cross-encoder reranking: reads (question, chunk) pairs together and re-sorts the hybrid candidate pool.

A cross-encoder is slower than the retrievers (one model pass per pair) but more precise, so it only
sees the top POOL chunks of the hybrid search; the best few go to the answering LLM.
Candidates (multilingual or French, open weights); load from models/<alias>/ when downloaded manually.
"""

from __future__ import annotations

import numpy as np

from aides_rag.retrieval.dense import local_dir, model_revision

RERANKERS = {
    "mminilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "camembert": "antoinelouis/crossencoder-camembert-base-mmarcoFR",
    "bge-reranker": "BAAI/bge-reranker-v2-m3",
}


class Reranker:
    def __init__(self, alias: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder  # heavy import, only when used

        self.alias = alias
        self.name = RERANKERS[alias]
        self.model = CrossEncoder(str(local_dir(alias) or self.name), max_length=max_length, device="cpu")
        self.revision = model_revision(self.name, alias)

    def scores(self, query: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=batch_size,
                                             show_progress_bar=False), dtype=np.float32)

    def rerank(self, query: str, candidates: list[int], texts: list[str],
               batch_size: int = 16) -> tuple[list[int], np.ndarray]:
        """Candidate chunk indices re-sorted by cross-encoder score (highest first), with their scores."""
        s = self.scores(query, [texts[i] for i in candidates], batch_size)
        order = np.argsort(-s, kind="stable")
        return [candidates[i] for i in order], s[order]
