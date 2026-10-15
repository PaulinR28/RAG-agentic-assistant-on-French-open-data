"""Checks on the chunking and the BM25 index."""

import json
from pathlib import Path

import numpy as np
import pytest

from aides_rag.index.chunks import SECTIONS, build_chunks, expand_placeholders, read_corpus
from aides_rag.retrieval.bm25 import BM25, analyze
from aides_rag.retrieval.eval_recall import evaluate

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data/processed/chunks.jsonl"


@pytest.fixture(scope="module")
def aids() -> list[dict]:
    return read_corpus(ROOT / "data/processed/aids.jsonl")


@pytest.fixture(scope="module")
def chunks(aids) -> list[dict]:
    return build_chunks(aids)


def test_committed_chunks_match_rebuild(chunks):
    assert read_corpus(CHUNKS) == chunks


def test_every_aid_chunked_once(aids, chunks):
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert {c["id_aid"] for c in chunks} == {a["id_aid"] for a in aids}
    for c in chunks:
        assert c["chunk_id"] == f"{c['id_aid']}-{c['chunk_index']}"
        assert 0 <= c["chunk_index"] < c["n_chunks"]


def test_chunk_content(chunks):
    for c in chunks:
        assert c["text"].startswith(f"Aide : {c['title']}")
        assert "%pme%" not in c["text"] and "%mini%" not in c["text"]
        assert 0 < c["approx_tokens"] <= 800, c["chunk_id"]


def test_no_text_lost(aids, chunks):
    """Start and end of every section survive chunking (whitespace-normalised: splitting rejoins lines)."""
    squash = " ".join
    by_aid: dict[str, str] = {}
    for c in chunks:
        by_aid[c["id_aid"]] = by_aid.get(c["id_aid"], "") + squash(c["body"].split()) + " "
    for aid in aids:
        for field, _ in SECTIONS:
            words = expand_placeholders(aid[field] or "").split()
            for probe in (squash(words[:6]), squash(words[-6:])):
                assert probe in by_aid[aid["id_aid"]], (aid["id_aid"], field)


def test_analyzer_is_accent_and_case_insensitive():
    assert analyze("Transition ÉCOLOGIQUE") == analyze("transition ecologique")
    assert analyze("les subventions") == analyze("subvention")


def test_bm25_finds_named_aid(chunks):
    index = BM25([c["text"] for c in chunks])
    top = [chunks[i]["id_aid"] for i, _ in index.search("Prêt Vert Bpifrance", k=5)]
    assert "10321" in top
    assert index.search("zzzz qwerty") == []


def test_bm25_recall_floor(aids):
    """Regression guard on the golden questions (dev set); measured 1.0 at commit time."""
    questions = [q for q in read_corpus(ROOT / "eval/golden.jsonl") if q["gold_docs"]]
    result = evaluate(aids, questions, stem=True, header=True, budget=1800)
    assert result["summary"]["recall@5"] >= 0.9
