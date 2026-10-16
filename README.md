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
DATA_CARD.md                  corpus provenance, selection rules, distributions, limitations
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

## Golden set format

```json
{"id": "q007", "category": "eligibility", "question": "...", "answer": "...",
 "gold_docs": ["12636"], "expected_behavior": "answer", "rationale": "...", "split": "dev"}
```

40 questions: `split` = `dev` (q001-q020, used to tune) or `heldout` (q021-q040, final scores only;
no aid appears in both splits).

- `category`: `factual`, `eligibility`, `geographic`, `deadline`, `exclusion`, `no_amount`, `out_of_corpus`, `prompt_injection`
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

All models run locally on CPU (demo machine: 16 GB RAM, Intel Core Ultra 5 135U, integrated graphics only).
Exact Hugging Face revisions are pinned; weights are never committed (`models/` is git-ignored).

| Role | Model | Revision | Status |
|---|---|---|---|
| Dense retriever | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | `5617a9f61b028005a4858fdac845db406aefb181` | **Chosen** |
| Dense (compared) | `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | Benchmarked |
| Dense (compared) | `intfloat/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` | Benchmarked |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | Under evaluation |
| Keyword | BM25 (Okapi, k1=1.5, b=0.75), French Snowball stemmer | `snowballstemmer` (see `uv.lock`) | In use |
| Answering LLM | to be chosen (Ollama) | | |

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

Benchmarks: `uv run python -m aides_rag.retrieval.bench_dense --model bge-m3`,
`uv run python -m aides_rag.retrieval.bench_rerank --reranker mminilm`, and the full runner
`uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --system hybrid`.
