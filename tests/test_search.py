"""Checks on the agent's search() contract: hybrid ranking, one hit per aid, metadata filters, BM25 fallback."""

import json
from pathlib import Path

import pytest

from aides_rag.retrieval import search as S
from aides_rag.retrieval.dense import MODELS, model_revision

ROOT = Path(__file__).resolve().parents[1]
needs_bge = pytest.mark.skipif(model_revision(MODELS["bge-m3"].name, "bge-m3") is None, reason="bge-m3 not downloaded")


def golden() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval/golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_bm25_fallback_one_hit_per_aid(monkeypatch):
    monkeypatch.setenv("AIDES_SEARCH", "bm25")
    hits = S.search("prêt vert Bpifrance transition écologique", k=5)
    assert 0 < len(hits) <= 5
    assert len({h.id_aid for h in hits}) == len(hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_region_filter_keeps_national_and_unresolved(monkeypatch):
    monkeypatch.setenv("AIDES_SEARCH", "bm25")
    chunks, _ = S._index()
    scope = {c["id_aid"]: (c["scope"], c["regions"]) for c in chunks}
    for h in S.search("subvention investissement", k=30, region="Bretagne"):
        sc, regions = scope[h.id_aid]
        assert sc in S.EVERYWHERE or not regions or "Bretagne" in regions


def test_open_on_filter_drops_closed_aids(monkeypatch):
    monkeypatch.setenv("AIDES_SEARCH", "bm25")
    for h in S.search("subvention investissement matériel", k=30, open_on="2026-10-16"):
        assert h.date_fin is None or h.date_fin >= "2026-10-16"


@needs_bge
def test_hybrid_search_matches_the_measured_default():
    """search() must rank aids exactly like run_eval --system hybrid (eval/retrieval_fusion.json, rrf_w3)."""
    from aides_rag.systems.hybrid import Hybrid

    system = Hybrid(k=10)
    for q in golden():
        assert [h.id_aid for h in S.search(q["question"], k=10)] == system.answer(q["question"])["retrieved_ids"], q["id"]


@needs_bge
def test_hybrid_filters_can_reach_any_aid():
    """Even a strict filter returns k hits: aids outside both top-50 lists are still ranked (dense order)."""
    hits = S.search("aide", k=10, natures=["Garantie"])
    assert len(hits) == min(10, 3)  # the corpus has 3 guarantee aids
