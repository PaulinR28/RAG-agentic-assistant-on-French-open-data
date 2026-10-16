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
