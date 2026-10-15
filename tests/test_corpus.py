"""Integrity checks on the committed corpus, its selection report and the golden set."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/aids.jsonl"
REPORT = ROOT / "data/processed/selection_report.json"
GOLDEN = ROOT / "eval/golden.jsonl"
RAW = ROOT / "data/raw"
THEMES = {"Transition écologique et énergétique", "Innovation"}
from aides_rag.ingest.select_subset import REGION_DEPTS, Territories

REGIONS = set(REGION_DEPTS)
ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
raw_needed = pytest.mark.skipif(not (RAW / "aides.csv").exists(), reason="raw data not present (git-ignored)")


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return pd.read_json(CORPUS, lines=True, dtype={"id_aid": str})


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_corpus_is_strict_json():
    def reject(token):
        raise ValueError(f"non-standard JSON constant {token}")

    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        json.loads(line, parse_constant=reject)


def test_count_matches_report(corpus, report):
    assert len(corpus) == report["n_selected"] == report["steps"][-1]["kept"]


def test_report_steps_chain(report):
    steps = report["steps"]
    for prev, cur in zip(steps, steps[1:]):
        assert cur["kept"] + cur["dropped"] == prev["kept"]


def test_output_hash_matches_report(report):
    assert hashlib.sha256(CORPUS.read_bytes()).hexdigest() == report["output_sha256"]


def test_ids_unique_and_numeric(corpus):
    assert corpus.id_aid.is_unique
    assert corpus.id_aid.str.fullmatch(r"\d+").all()


def test_every_aid_in_theme(corpus):
    assert corpus.project_categories.map(lambda c: bool(THEMES & set(c))).all()


def test_text_present(corpus):
    assert (corpus.title.str.len() > 0).all()
    assert ((corpus.objet.str.len() > 0) | (corpus.operations.str.len() > 0)).all()
    assert (corpus.n_chars >= 600).all()


def test_no_exact_duplicates(corpus):
    assert not corpus.duplicated(["title", "objet", "montant"]).any()


def test_geography(corpus):
    assert set(corpus.scope) <= {"national", "european", "territorial"}
    assert set(corpus.regions.explode().dropna()) <= REGIONS
    has_region = corpus.regions.map(len) > 0
    assert (has_region == corpus.region_source.notna()).all()
    assert corpus.loc[corpus.region_source == "financeur_postcode", "scope"].eq("territorial").all()


def test_dates(corpus):
    for col in ("date_fin", "aid_validation"):
        values = corpus[col].dropna()
        assert values.map(lambda v: bool(ISO.fullmatch(v))).all(), col
    assert corpus.aid_validation.notna().all()


def test_source_urls_well_formed(corpus):
    for url in corpus.source_urls.explode().dropna():
        parsed = urlparse(url)
        assert parsed.scheme in ("http", "https") and "." in parsed.netloc, url


def test_funders_resolved(corpus):
    assert (corpus.financeurs.map(len) == corpus.financeur_ids.map(len)).all()
    assert (corpus.financeurs.map(len) > 0).all()


@raw_needed
def test_region_resolution_overrides_bad_source_links():
    ter = Territories(RAW)
    # Aisne (26), Oise (85), Somme (105) are linked to Occitanie in territoires.csv; must be Hauts-de-France.
    assert ter.regions(["26", "85", "105"]) == ["Hauts-de-France"]
    assert ter.regions(["21"]) == ["Hauts-de-France"]  # pre-2016 Picardie id
    assert ter.regions(["56"]) == ["Occitanie"]  # Haute-Garonne
    assert ter.regions(["121"]) == ["Outre-mer"]  # Guadeloupe
    assert ter.regions(["50006"]) == ["Île-de-France"]
    assert ter.regions_from_postcode("60000") == ["Hauts-de-France"]
    assert ter.regions_from_postcode("20000") == ["Corse"]


@raw_needed
def test_ids_exist_in_raw(corpus):
    raw = pd.read_csv(RAW / "aides.csv", sep=";", encoding="cp1252", dtype=str, keep_default_na=False)
    assert set(corpus.id_aid) <= set(raw.id_aid)


@raw_needed
def test_rebuild_is_deterministic(tmp_path, report):
    from aides_rag.ingest import select_subset as sel

    selected, rebuilt_report, proj_label = sel.select(RAW)
    assert rebuilt_report["raw_sha256"] == report["raw_sha256"], "raw files differ from the committed build"
    out = tmp_path / "aids.jsonl"
    sel.write_jsonl(sel.build_records(selected, RAW, proj_label), out)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == report["output_sha256"]


# ---------------------------------------------------------------- golden set


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_golden_schema(golden):
    required = {"id", "question", "answer", "gold_docs", "category", "expected_behavior", "rationale"}
    for q in golden:
        assert required <= set(q), q.get("id")
        assert isinstance(q["gold_docs"], list)
    assert len({q["id"] for q in golden}) == len(golden)


def test_golden_docs_in_corpus(golden, corpus):
    ids = set(corpus.id_aid)
    for q in golden:
        assert set(q["gold_docs"]) <= ids, q["id"]


def test_golden_unanswerable_have_no_docs(golden):
    for q in golden:
        if q["expected_behavior"] in ("abstain", "refuse_injection"):
            assert q["category"] in ("out_of_corpus", "prompt_injection"), q["id"]
        if q["category"] == "out_of_corpus":
            assert q["gold_docs"] == [], q["id"]
