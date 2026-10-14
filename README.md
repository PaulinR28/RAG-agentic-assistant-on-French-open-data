# RAG & agentic assistant on French open data: green & innovation business aids

An assistant that answers **"Which public funding can my company get?"** for French SMEs with a
green-transition or innovation project. It runs over 504 public aid schemes from the
[data.gouv.fr "Aides entreprises" dataset](https://www.data.gouv.fr/datasets/aides-entreprises)
(Institut Supérieur des Métiers, Licence Ouverte / Open Licence). It uses
hybrid retrieval with reranking, answers with citations, and runs locally.

> Status: **Phase 1, foundations** (corpus, data card, golden set). Retrieval, agent and evaluation runner come next.

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create .venv and install dependencies
cp .env.example .env                      # local secrets (never committed)

# Rebuild the corpus (needs the raw CSVs in data/raw/, see data/raw/README.md)
uv run python -m aides_rag.ingest.select_subset --raw data/raw --out data/processed

uv run pytest                             # corpus + golden-set integrity checks
uv run python -m aides_rag.ingest.check_sources   # optional, needs network: source URL check
```

Evaluation (Phase 2 onwards):

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

## Repository layout

```
DATA_CARD.md                  corpus provenance, selection rules, distributions, limitations
data/raw/                     raw CSVs (git-ignored)
data/processed/aids.jsonl     the 504-aid corpus (committed, ~1.9 MB)
data/processed/selection_report.json   step counts, raw + output checksums
data/processed/source_check.json       last URL reachability check
src/aides_rag/ingest/         corpus building (select_subset.py, check_sources.py)
eval/golden.jsonl             golden questions (id, question, answer, gold_docs, category, expected_behavior, rationale)
tests/                        pytest integrity checks
```

## Golden set format

```json
{"id": "q007", "category": "eligibility", "question": "...", "answer": "...",
 "gold_docs": ["12636"], "expected_behavior": "answer", "rationale": "..."}
```

- `category`: `factual`, `eligibility`, `geographic`, `deadline`, `out_of_corpus`, `prompt_injection`
- `expected_behavior`: `answer`, `abstain`, `refuse_injection`, `answer_ignore_injection`
- `gold_docs`: `id_aid` values from `data/processed/aids.jsonl`; empty when the corpus has no answer.
- Questions that depend on today's date state the date explicitly.

## Git workflow

- `main` is always runnable. No direct pushes after the initial skeleton.
- One short-lived branch per task: `feat/…`, `fix/…`, `eval/…`, `docs/…`.
- Pull request into `main`, with **one reviewer** and `uv run pytest` passing before merge.
- Never commit raw data, model weights or secrets (`.env`).
- Code freeze: **2026-10-20 23:59**, tag `v1.0`.

## Models

To be fixed in Phase 2, with exact names and versions. Demo machine constraint: 16 GB RAM, Intel Core
Ultra 5 135U, integrated graphics only (CPU inference).
