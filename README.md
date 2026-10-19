# RAG & agentic assistant on French open data: green & innovation business aids

An assistant that answers **"Which public funding can my company get?"** for French SMEs with a
green-transition or innovation project. It runs over 504 public aid schemes from the
[data.gouv.fr "Aides entreprises" dataset](https://www.data.gouv.fr/datasets/aides-entreprises)
(Institut Supérieur des Métiers, Licence Ouverte / Open Licence). It uses
hybrid retrieval (BM25 + bge-m3 embeddings), answers with citations, and runs locally.

> Status: **Phase 1, foundations** (corpus, data card, golden set). Retrieval, agent and evaluation runner come next.

## 1. Results

<!-- TODO: from eval/summary.json. Held-out split (q021–q040), demo machine, CPU only. Full tables: EVALUATION.md §4 -->

| System | Recall@5 | MRR | Correct | Behaviour OK | p50 latency (s) | € / question |
|---|---|---|---|---|---|---|
| `bm25` (baseline, no LLM) | 0.875 | 0.81 | n/a | n/a | < 0.01 | 0 |
| `hybrid` (no LLM, retrieval used by `rag` / `agent`) | **1.00** | **0.92** | n/a | n/a | 0.22 | 0 |
| `rag` | TODO | TODO | TODO | TODO | TODO | TODO |
| **`agent` (final)** | TODO | TODO | TODO | TODO | TODO | TODO |

- Hybrid search (BM25 + bge-m3, dense weight 3) over BM25 alone, held-out: Recall@1 0.75 → 0.88, Recall@5 0.88 → 1.00, for ~0.2 s per question on CPU. Cross-encoder reranking was measured and left off by default: no held-out gain for +20 s per question (EVALUATION.md §5, E5–E6).
- TODO: quality vs latency between `rag` and `agent`.
- TODO: main remaining failure mode.

Demo recording: <!-- TODO: link -->

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create .venv and install dependencies
cp .env.example .env                      # local secrets (never committed)

# Rebuild the corpus (needs the raw CSVs in data/raw/, see data/raw/README.md)
uv run python -m aides_rag.ingest.select_subset --raw data/raw --out data/processed

uv run python -m aides_rag.index.chunks   # aids.jsonl -> chunks.jsonl (header + body, ~300-700 tokens)
uv run python -m aides_rag.retrieval.eval_recall  # BM25 Recall@k / MRR on the golden set -> eval/retrieval_bm25.json

uv run pytest                             # corpus, chunking, BM25 and golden-set checks
uv run python -m aides_rag.ingest.check_sources   # optional, needs network: source URL check
```

Evaluation:

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/   # --system bm25 (default)
```

It writes `eval/results.jsonl` (one line per question: answer, cited sources, retrieved aid ids,
tokens in/out, latency, tool calls, stop reason) and `eval/summary.json` (Recall@k, MRR, latency,
tokens). Every system implements the same `answer(question) -> dict` contract, documented in
`src/aides_rag/systems/__init__.py`; add a new one there and compare with `--system <name>`.

## Repository layout

```
DATA_CARD.md                  corpus provenance, selection, distributions, cleaning, chunking, limitations
EVALUATION.md                 golden set, metrics, results, experiments, error analysis, risks
data/raw/                     raw CSVs (git-ignored)
data/processed/aids.jsonl     the 504-aid corpus (committed, ~1.9 MB)
data/processed/selection_report.json   step counts, raw + output checksums
data/processed/source_check.json       last URL reachability check
data/processed/chunks.jsonl   865 retrieval chunks (metadata header + body), rebuilt from aids.jsonl
src/aides_rag/ingest/         corpus building (select_subset.py, check_sources.py)
src/aides_rag/index/          chunking (chunks.py)
src/aides_rag/retrieval/      BM25 with French analyzer (bm25.py), retrieval evaluation (eval_recall.py)
src/aides_rag/systems/        answering systems behind one contract (bm25_only.py: retrieval only, no LLM)
src/aides_rag/evaluation/     evaluation runner (run_eval.py), called by eval/run_eval.py
eval/run_eval.py              evaluation entry point -> eval/results.jsonl, eval/summary.json
eval/retrieval_bm25.json      last retrieval evaluation (variants, per-question ranks)
eval/golden.jsonl             golden questions (id, question, answer, gold_docs, category, expected_behavior, rationale)
tests/                        pytest integrity checks
```

## Models

To be fixed in Phase 2, with exact names and versions. Demo machine constraint: 16 GB RAM, Intel Core
Ultra 5 135U, integrated graphics only (CPU inference).

Demo machine: 16 GB RAM, Intel Core Ultra 5 135U, integrated graphics only (CPU inference).
Library versions are pinned in `uv.lock`.

| Role | Model | Exact version / tag | Runtime |
|---|---|---|---|
| Embeddings | `BAAI/bge-m3` | HF revision `5617a9f61b028005a4858fdac845db406aefb181` | sentence-transformers, CPU |
| Reranker (option, off by default) | `BAAI/bge-reranker-v2-m3` | HF revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | sentence-transformers, CPU |
| LLM (local) | TODO | TODO (tag + quantisation) | Ollama TODO |
| LLM (API, optional) | TODO | TODO (dated snapshot) | API |
| Judge | TODO | TODO | TODO |

## Structured outputs and tracking

Every answer is validated against a schema (`TODO: path`). On invalid output: TODO (retry N times, then abstain).
Cited ids are checked against retrieved ids.

```json
{"status": "answer | abstain | refuse_injection", "answer": "...",
 "aids": [{"id_aid": "12636", "name": "...", "why_relevant": "..."}], "citations": ["12636"]}
```

Every system implements the same `answer(question) -> dict` contract, documented in
`src/aides_rag/systems/__init__.py`. Each line of `eval/results.jsonl` records tokens in/out
(TODO: source), end-to-end latency, every tool call with its arguments and latency, the stop reason
(TODO: values) and the cost (local: 0 € API cost; API: tokens × price on TODO date).

## 6. Evaluation and residual risks

40 golden questions in `eval/golden.jsonl`, across 8 categories (factual, eligibility, geographic, deadline,
exclusion, no amount, out of corpus, prompt injection). `dev` (q001–q020) is used to tune and `heldout`
(q021–q040) for final scores only; no aid appears in both. Metrics: Recall@k, MRR, judged correctness,
citation precision, abstention / injection behaviour, latency, tokens, cost.
Details, experiments and error analysis: [`EVALUATION.md`](EVALUATION.md).

Main residual risks (full table: `EVALUATION.md` §7):
- **Stale data:** 2026-10-13 snapshot of a weekly-updated source; deadlines and amounts may change.
- **Region coverage:** 204 of 432 territorial aids have no machine-readable region.
- **Not advice:** eligibility answers are indicative; the funder decides.
- **Small test set:** 20 held-out questions, so one question = 5 points.
- TODO: hallucination / injection rate from the final run.

**Bonuses:** UI TODO · vector DB with metadata filters TODO · date handling TODO · local vs API comparison TODO

## Team

<!-- TODO: everyone speaks at the defence -->
| Member | Main contributions | Defence part |
|---|---|---|
| Izan-Gaël | TODO | TODO |
| Oussama | TODO | TODO |
| Paulin | TODO | TODO |
| Léopold | TODO | TODO |

## Git workflow

- `main` is always runnable. No direct pushes after the initial skeleton.
- One short-lived branch per task: `feat/…`, `fix/…`, `eval/…`, `docs/…`.
- Pull request into `main`, with **one reviewer** and `uv run pytest` passing before merge.
- Never commit raw data, model weights or secrets (`.env`).
- Code freeze: **2026-10-20 23:59**, tag `v1.0`.

---

*Data: Institut Supérieur des Métiers – aides-entreprises.fr, via data.gouv.fr, Licence Ouverte v1.0.
Snapshot received 2026-10-13 (latest record update 2026-10-09).*
