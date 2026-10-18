"""Checks on the evaluation runner (eval/run_eval.py)."""

import json
from pathlib import Path

import pytest

from aides_rag.evaluation.run_eval import main
from aides_rag.systems import OUTPUT_KEYS, load_system

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval/golden.jsonl"


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    summary = main(["--questions", str(GOLDEN), "--out", str(out), "--system", "bm25"])
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    return summary, rows


def test_one_row_per_question_with_required_fields(full_run):
    summary, rows = full_run
    n_golden = sum(1 for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip())
    assert len(rows) == summary["n_questions"] == n_golden
    for row in rows:
        for key in (*OUTPUT_KEYS, "id", "latency_ms", "error"):
            assert key in row
        assert row["error"] is None
        assert row["latency_ms"] >= 0


def test_recall_matches_retrieval_eval(full_run):
    """The runner and retrieval/eval_recall.py must agree on the same BM25 index."""
    summary, _ = full_run
    reference = json.loads((ROOT / "eval/retrieval_bm25.json").read_text(encoding="utf-8"))
    default = reference["variants"][reference["default"]]
    for split in ("dev", "heldout"):
        expected = default[split]["summary"]
        got = summary["retrieval_by_split"][split]
        assert got["n_questions"] == reference["questions"][split]
        for k in (1, 5, 10):
            assert got[f"recall@{k}"] == expected[f"recall@{k}"], (split, k)


def test_failing_question_is_recorded_not_raised(tmp_path, monkeypatch):
    def boom(self, question):
        raise RuntimeError("model down")

    monkeypatch.setattr("aides_rag.systems.bm25_only.Bm25Only.answer", boom)
    summary = main(["--questions", str(GOLDEN), "--out", str(tmp_path), "--limit", "2"])
    assert len(summary["errors"]) == 2


def test_unknown_system_rejected():
    with pytest.raises(ValueError):
        load_system("does-not-exist")


@pytest.mark.skipif(not (ROOT / "models/bge-m3/REVISION").exists(), reason="bge-m3 not downloaded")
def test_hybrid_system_matches_fusion_benchmark(tmp_path):
    """run_eval --system hybrid and retrieval/bench_fusion.py must agree (BM25 + bge-m3, dense weighted 3)."""
    summary = main(["--questions", str(GOLDEN), "--out", str(tmp_path), "--system", "hybrid"])
    reference = json.loads((ROOT / "eval/retrieval_fusion.json").read_text(encoding="utf-8"))["quality"]
    for split in ("dev", "heldout"):
        expected = reference[split]["rrf_w3"]["summary"]
        for k in (1, 5, 10):
            assert summary["retrieval_by_split"][split][f"recall@{k}"] == expected[f"recall@{k}"], (split, k)
