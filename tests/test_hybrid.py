"""Checks on rank fusion (no model needed) and, when a model is cached locally, on dense search."""

import numpy as np
import pytest

from aides_rag.retrieval.dense import MODELS, model_revision
from aides_rag.retrieval.hybrid import rank_aids, rrf


def test_rank_aids_keeps_best_chunk_per_aid():
    scores = np.array([0.2, 0.9, 0.5, 0.0])
    chunk_aids = np.array(["a", "b", "a", "c"])
    assert rank_aids(scores, chunk_aids) == ["b", "a", "c"]
    assert rank_aids(scores, chunk_aids, positive_only=True) == ["b", "a"]


def test_rrf_rewards_agreement_and_uses_ranks_only():
    bm25 = ["x", "y", "z"]
    dense = ["y", "w", "x"]
    fused = rrf([bm25, dense])
    assert fused[0] == "y"  # 2nd + 1st beats 1st + 3rd
    assert set(fused) == {"x", "y", "z", "w"}
    assert rrf([bm25, dense], depth=1) == ["x", "y"]  # tie broken by id, deterministic


@pytest.mark.skipif(model_revision(MODELS["e5-small"].name) is None, reason="e5-small not downloaded")
def test_dense_finds_paraphrase(tmp_path):
    from aides_rag.retrieval.dense import DenseIndex

    texts = ["Subvention pour l'achat de vélos cargos électriques de livraison",
             "Crédit d'impôt pour les dépenses de recherche et développement",
             "Aide à la rénovation énergétique des bâtiments"]
    index = DenseIndex("e5-small", texts, cache_dir=tmp_path)
    assert int(np.argmax(index.scores("financer un triporteur à assistance électrique"))) == 0
    assert int(np.argmax(index.scores("isoler la toiture de mon atelier"))) == 2


def test_hybrid_chunks_fuses_both_retrievers():
    from aides_rag.retrieval.hybrid import aids_in_order, hybrid_chunks

    bm25 = np.array([0.0, 5.0, 1.0, 0.0])  # chunk 1 first, chunk 0 and 3 unmatched
    dense = np.array([0.9, 0.1, 0.5, 0.8])  # chunk 0 first
    # bm25 ranks [1, 2]; dense ranks [0, 3, 2, 1] -> 1: 1/61+1/64, 2: 1/62+1/63, 0: 1/61, 3: 1/62
    assert hybrid_chunks(bm25, dense, depth=4) == [1, 2, 0, 3]
    assert aids_in_order([3, 0, 2], np.array(["a", "a", "b", "c"])) == ["c", "a", "b"]


@pytest.mark.skipif(model_revision(MODELS["bge-m3"].name, "bge-m3") is None, reason="bge-m3 not downloaded")
def test_no_chunk_is_truncated_by_the_dense_model():
    """Real token counts (not the chars/token estimate): every chunk fits bge-m3's window, and the
    bge-reranker window fits the longest golden question + the longest chunk."""
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from aides_rag.retrieval.dense import local_dir
    from aides_rag.retrieval.reranker import MAX_LENGTH

    root = Path(__file__).resolve().parents[1]
    texts = [json.loads(line)["text"] for line in (root / "data/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = [json.loads(line)["question"] for line in (root / "eval/golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tok = AutoTokenizer.from_pretrained(str(local_dir("bge-m3") or MODELS["bge-m3"].name))
    longest_chunk = max(len(tok(t)["input_ids"]) for t in texts)
    longest_question = max(len(tok(q)["input_ids"]) for q in questions)
    assert longest_chunk <= MODELS["bge-m3"].max_seq_length
    assert longest_chunk + longest_question <= MAX_LENGTH["bge-reranker"]
