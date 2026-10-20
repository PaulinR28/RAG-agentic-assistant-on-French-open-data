# RAG & agentic assistant on French open data: green & innovation business aids

An assistant that answers **"Which public funding can my company get?"** for French SMEs with a
green-transition or innovation project. It runs over 504 public aid schemes from the
[data.gouv.fr "Aides entreprises" dataset](https://www.data.gouv.fr/datasets/aides-entreprises)
(Institut Supérieur des Métiers, Licence Ouverte / Open Licence). It uses
hybrid retrieval (BM25 + bge-m3 embeddings), answers with citations, and runs locally.

> Status: corpus, hybrid retrieval, generation (`rag`, `agent`), evaluation runner, LLM judge and UI are built;
> final held-out runs done on 2026-10-19 (commit `ca93aac`). Code freeze: 2026-10-20 23:59, tag `v1.0`.

## Results

Held-out split (q021–q040), run once after the final system was fixed. Correct = judged correct / partial / wrong out of 20. Full tables: EVALUATION.md §4.
LLM runs on a MacBook Air M3 (16 GB, `eval/final/RUN_INFO.txt`); on the demo laptop (Intel CPU only) local answers
take about 2.5× longer (dev p50 ~59 s).

| System | Recall@5 | MRR | Correct | Behaviour OK | p50 latency (s) | USD / question |
|---|---|---|---|---|---|---|
| `bm25` (baseline, no LLM) | 0.875 | 0.81 | n/a | n/a | < 0.01 | 0 |
| `hybrid` (no LLM, retrieval used by `rag` / `agent`) | **1.00** | **0.92** | n/a | n/a | 0.22 | 0 |
| **`rag`, Qwen3 4B local (final)** | 1.00 | 0.92 | 11 / 3 / 6 | 17 / 20 | 23.1 | 0 |
| `agent`, Qwen3 4B local | 0.88 | 0.78 | 11 / 3 / 6 | 19 / 20 | 26.8 | 0 |
| `rag`, Claude Haiku 5.5 (API) | 1.00 | 0.92 | 15 / 2 / 3 | 19 / 20 | 2.1 | 0.0007 |
| `agent`, Claude Haiku 5.5 (API) | 0.94 | 0.88 | 16 / 2 / 2 | 19 / 20 | 4.7 | 0.0017 |

- Hybrid search (BM25 + bge-m3, dense weight 3) over BM25 alone, held-out: Recall@1 0.75 → 0.88, Recall@5 0.88 → 1.00, for ~0.2 s per question on CPU. Cross-encoder reranking was measured and left off by default: no held-out gain for +20 s per question (EVALUATION.md §5, E5–E6).
- Local `agent` costs three times the tokens of `rag` and is slower, for no gain; with Haiku, `agent` gains one question over `rag`. `rag` stays the final local system.
- Main remaining failure: the 4B model misreads conditions (thresholds, dates, exclusions) in the right aid. For every wrong answer of the final `rag` system the gold aid was in the top 5 (local `agent`: all but q026); the same pipeline with Claude Haiku 5.5 gets 15–16 / 20. Out-of-corpus questions: 3 / 3 abstained by all systems; 0 errors, 20 / 20 valid outputs.

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
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/   # --system rag (default)
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --system agent   # or bm25, hybrid, hybrid_rerank
```

Systems: `bm25`, `hybrid`, `hybrid_rerank` (retrieval only, no LLM), `rag` and `agent` (answer with the LLM).
`rag` and `agent` retrieve with `hybrid`. The answering model is local by default; run the same system with an
API model through OpenRouter, e.g. `LLM_BACKEND=openrouter LLM_MODEL=anthropic/claude-haiku-5.5` (key in `.env`).

Local model: install [Ollama](https://ollama.com), then `ollama pull qwen3:4b-instruct-2507-q4_K_M` (the
**instruct** tag: plain `qwen3:4b` always reasons first and is much slower). One question with its trace:
`uv run python -m aides_rag.agent "Quel est le taux du CIR ?"` (`--mode rag` or `agent`).

It writes `eval/results.jsonl` (one line per question: answer, cited sources, retrieved aid ids,
tokens in/out, latency, tool calls, stop reason, cost, config and each LLM call) and `eval/summary.json`
(Recall@k, MRR, latency, tokens, total cost, models, stop reasons, errors). Every system implements the same `answer(question) -> dict` contract, documented in
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
src/aides_rag/retrieval/      BM25 (bm25.py), bge-m3 (dense.py), weighted RRF (hybrid.py), reranker option
                              (reranker.py), search() with filters and place hints (search.py, geo.py), benchmarks
src/aides_rag/llm/            one chat() client for Ollama (local) and OpenRouter (API)
src/aides_rag/agent/          answer schema + checks (schema.py), tools (tools.py), rag / agent loop (agent.py),
                              per-answer trace (trace.py), CLI (python -m aides_rag.agent)
src/aides_rag/systems/        the systems scored by run_eval, behind one contract (bm25, hybrid, rag, agent...)
src/aides_rag/evaluation/     evaluation runner (run_eval.py), called by eval/run_eval.py
eval/run_eval.py              evaluation entry point -> eval/results.jsonl, eval/summary.json
eval/retrieval_*.json         retrieval benchmarks (BM25 variants, dense models, fusion, rerankers; per-question ranks)
eval/experiments/             generation experiments (e.g. e9_full_cards_geo)
indexes/                      bge-m3 chunk vectors (committed, pinned to the model revision)
eval/golden.jsonl             golden questions (id, question, answer, gold_docs, category, expected_behavior, rationale)
tests/                        pytest integrity checks
```

## Models

All models run locally on CPU (demo machine: 16 GB RAM, Intel Core Ultra 5 135U, integrated graphics only).
Exact Hugging Face revisions are pinned; weights are never committed (`models/` is git-ignored).

| Role | Model | Revision | Status |
|---|---|---|---|
| Dense retriever | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | `5617a9f61b028005a4858fdac845db406aefb181` | **Chosen** |
| Dense (compared) | `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | Benchmarked |
| Dense (compared) | `intfloat/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` | Benchmarked |
| Reranker (option) | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | Built, **off by default** (`--system hybrid_rerank`) |
| Reranker (compared) | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | Rejected: lowers held-out Recall@1 |
| Keyword | BM25 (Okapi, k1=1.5, b=0.75), French Snowball stemmer | `snowballstemmer` (see `uv.lock`) | In use |
| Fusion | Reciprocal Rank Fusion, k=60, **dense weight 3 / BM25 1** | (code) | In use, picked on dev |
| Answering LLM (local, default) | `qwen3:4b-instruct-2507-q4_K_M` (Ollama tag; Q4_K_M quantisation) | Ollama 0.40.4, `temperature=0`, `num_ctx=8192` | **In use** |
| Answering LLM (API, comparison) | `anthropic/claude-haiku-5.5` via OpenRouter | TODO (dated snapshot) | Optional (`LLM_BACKEND=openrouter`) |
| Answering LLM (API, comparison) | Qwen3-14B via OpenRouter | TODO (exact slug) | Optional (`LLM_MODEL=<slug>`) |
| Judge | `anthropic/claude-sonnet-5.5` via OpenRouter | temperature 0, reasoning `low` | **In use** (`run_eval.py --judge`); 97.5 % agreement with our hand labels (20 dev questions × 2 runs = 40 answers; held-out never used) |

Library versions are pinned in `uv.lock`.

**Why bge-m3** (`eval/retrieval_dense.json`, dense retrieval alone):

| Model | Size | Recall@1 dev / held-out | Recall@5 dev / held-out | Query latency | Peak RAM |
|---|---|---|---|---|---|
| e5-small | 0.5 GB | 0.74 / 0.75 | 0.97 / 1.00 | 39 ms | 1.6 GB |
| e5-base | 1.1 GB | 0.74 / 0.63 | 1.00 / 1.00 | 86 ms | 2.4 GB |
| **bge-m3** | 2.3 GB | **0.91 / 0.88** | 0.97 / 1.00 | 219 ms | 2.6 GB |

Chosen on dev; held-out confirms. bge-m3 is also the only one whose top similarity separates answerable
from out-of-corpus questions (min 0.617 vs max 0.613, on 3 out-of-corpus questions only: a lead for
abstention, not a rule yet). The chunk vectors (`indexes/bge-m3-*.npy`, 3.5 MB) are committed, since
embedding the 865 chunks takes ~35 min on this CPU.

Get the models (resumable, pinned revision):

```bash
bash scripts/download_model.sh bge-m3 BAAI/bge-m3 5617a9f61b028005a4858fdac845db406aefb181 \
  config.json config_sentence_transformers.json modules.json sentence_bert_config.json \
  special_tokens_map.json tokenizer_config.json sentencepiece.bpe.model tokenizer.json \
  1_Pooling/config.json pytorch_model.bin
bash scripts/download_model.sh mminilm cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  1427fd652930e4ba29e8149678df786c240d8825 config.json special_tokens_map.json \
  tokenizer_config.json sentencepiece.bpe.model tokenizer.json model.safetensors
```

**Ranking: hybrid with dense weighted 3, reranker off by default** (`eval/retrieval_fusion.json`,
`eval/retrieval_rerank.json`; Recall@1 dev / held-out, extra time per question on this CPU):

| Ranking | Recall@1 | Recall@5 | Extra time |
|---|---|---|---|
| Hybrid, equal weights | 0.85 / 0.75 | 1.00 / 0.94 | 0 |
| **Hybrid, dense weight 3** (default) | **0.91 / 0.88** | 0.97 / **1.00** | 0 |
| + mMiniLMv2 reranker, pool 10 | 0.85 / 0.50 | 0.94 / 0.94 | +2.6 s |
| + bge-reranker-v2-m3, pool 10 (1,024 tokens) | 0.91 / 0.75 | 1.00 / 0.94 | +20.5 s |

The weight was picked on dev with a rule fixed beforehand (Recall@1, then MRR, then the simplest); held-out
was only checked afterwards. The cross-encoder rerankers bring no held-out gain on this corpus and, on CPU,
cost 2.6 to 19 s per question, so reranking stays available as an option (`hybrid_rerank`: weighted
fusion, top 10 re-sorted by bge-reranker-v2-m3) but is off in the default pipeline.

**Input length (real tokenizer counts, not estimates).** Chunks are 390 tokens median, 636 max. bge-m3 reads
1,024, so no chunk is truncated (tested). The e5 models read 512, so 14% of chunks (123) lost their tail in
their benchmark, which may partly explain their lower scores. The reranker benchmarks above ran at 512 tokens for
question + chunk, which truncates up to 38% of pairs; bge-reranker now uses 1,024 (its limit is 8,192).
Re-measured at 1,024: identical Recall (0.91 / 0.75 Recall@1, 1.00 / 0.94 Recall@5), 20.5 s per question, so
truncation was not what held it back.

Benchmarks: `uv run python -m aides_rag.retrieval.bench_dense --model bge-m3`,
`uv run python -m aides_rag.retrieval.bench_fusion`,
`uv run python -m aides_rag.retrieval.bench_rerank --reranker bge-reranker --pools 10 --batch-size 4`, and the full runner
`uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --system hybrid`.

## Structured outputs and tracking

The model must reply with this JSON, enforced at generation time and parsed with Pydantic
(`src/aides_rag/agent/schema.py`):

```json
{"answer": "...", "citations": ["12636"], "abstained": false}
```

Checks after parsing: every citation is an aid retrieved for this question, an answer that does not abstain
cites at least one aid, and an abstention cites none. On a violation the model gets **one retry** with the error
message; if that fails too, the row ends with `stop_reason = "invalid_output"` (no forced abstention).
Retrieved aids and the question are passed to the model as data, never as instructions.

Each line of `eval/results.jsonl` (`src/aides_rag/agent/trace.py`) records:
- **tokens in/out**: Ollama `prompt_eval_count` / `eval_count`; OpenRouter `usage`;
- **latency**: end to end, plus per LLM call and per tool call;
- **tool calls**: name, arguments, returned aid ids, latency, and whether the code forced the call;
- **stop reason**: `final_answer`, `abstained`, `max_tool_calls`, `invalid_output` or `error`;
- **cost in USD**: the cost billed by OpenRouter (`usage.cost`), not tokens × a price list; 0 for the local model.

## Evaluation and residual risks

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
- **Injection:** the direct injection (q039) is refused by all systems; with the hidden one (q040), the local model ignores it but abstains. No cited id outside the retrieved aids (blocked by the validator).

**Bonuses:** date handling: deadline status computed in code for a given day (`search_aids(open_on=…)`) ·
local vs API comparison: dev and held-out (`EVALUATION.md` §4–5) · UI: Streamlit (`uv run streamlit run src/aides_rag/ui/app.py`) ·
vector database: no (in-memory BM25 + bge-m3 index, with region and aid-type filters; deadline status is
annotated, not filtered, so a just-closed aid stays visible).

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
