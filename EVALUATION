# Evaluation

How we know the assistant works, how well, at what cost, and where it fails.

Course doctrine: no component counts as working until it is scored on the held-out golden set with
cost and latency tracked. Every design decision below was tested on the **dev** split; the **held-out**
split was run once per final system, for reporting only.

<!-- TODO: date and commit of the final runs -->
Final runs: TODO date, commit `TODO`, demo machine (16 GB RAM, Intel Core Ultra 5 135U, CPU only).

---

## 1. Golden set

File: `eval/golden.jsonl`, 40 questions, one JSON object per line.

```json
{"id": "q007", "category": "eligibility", "question": "...", "answer": "...",
 "gold_docs": ["12636"], "expected_behavior": "answer", "rationale": "...", "split": "dev"}
```

| Field | Content |
|---|---|
| `id` | `q001`–`q040` |
| `question` | Question in French, as an SME owner would ask it |
| `answer` | Reference answer, written from the gold aid(s) |
| `gold_docs` | `id_aid` values from `data/processed/aids.jsonl` that contain the answer; empty when the corpus has no answer |
| `category` | `factual`, `eligibility`, `geographic`, `deadline`, `exclusion`, `no_amount`, `out_of_corpus`, `prompt_injection` |
| `expected_behavior` | `answer`, `abstain`, `refuse_injection`, `answer_ignore_injection` |
| `rationale` | Why this answer and these gold docs |
| `split` | `dev` (q001–q020, used to tune) or `heldout` (q021–q040, final scores only) |

**Design rules**
- No aid appears in both splits, so tuning on dev cannot leak into held-out scores.
- Questions that depend on today's date state the date explicitly.
- `pytest` checks the schema, that every `gold_docs` id exists in the corpus, and that out-of-corpus questions have no docs.
- <!-- TODO: how questions were written (who, from which aids, how many reviewed by a second member) -->

**Composition** <!-- TODO: fill counts -->

| Category | What it tests | dev | heldout |
|---|---|---:|---:|
| factual | a fact stated in one aid (amount, funder, rate) | TODO | TODO |
| eligibility | can company X get aid Y (size, age, sector) | TODO | TODO |
| geographic | aids available in a given region | TODO | TODO |
| deadline | dates, "still open on <date>" | TODO | TODO |
| exclusion | conditions that rule a company out | TODO | TODO |
| no_amount | aid exists but the corpus gives no amount: must not invent one | TODO | TODO |
| out_of_corpus | no answer in the corpus: must abstain | TODO | TODO |
| prompt_injection | injected instruction in the question: refuse or ignore it | TODO | TODO |
| **Total** | | **20** | **20** |

---

## 2. Systems compared

All systems implement the same `answer(question) -> dict` contract (`src/aides_rag/systems/__init__.py`)
and are run with:

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --system <name>
```

<!-- TODO: adjust to the systems actually built -->
| System | Retrieval | Generation | Notes |
|---|---|---|---|
| `bm25` | BM25, French analyzer | none | baseline |
| `hybrid` | BM25 + `bge-m3`, weighted RRF (k = 60, dense weight 3) | none | retrieval of `rag` / `agent` |
| `hybrid_rerank` | hybrid, top 10 chunks re-sorted by `bge-reranker-v2-m3` | none | option, off by default |
| `rag` | hybrid, top-TODO chunks | TODO local LLM, one call | |
| `agent` | tools over hybrid + metadata filters | TODO local LLM, up to TODO tool calls | final system |
| `agent_api` | same as `agent` | TODO API model | optional comparison |

---

## 3. Metrics

| Metric | Definition | Computed on |
|---|---|---|
| Recall@k (k = 1, 5, 10) | share of questions with at least one gold doc in the top k retrieved aids | questions with non-empty `gold_docs` |
| MRR | mean of 1 / rank of the first gold doc (0 if absent from the top 10) | same |
| Correctness | TODO: judged `correct` / `partial` / `wrong` against the reference answer | `expected_behavior = answer` |
| Citation precision | share of cited ids that are gold docs or support the answer | answers with citations |
| Hallucinated citations | cited ids that were not retrieved | all answers |
| Behaviour accuracy | abstains, refuses or ignores the injection as expected | `abstain`, `refuse_injection`, `answer_ignore_injection` |
| Valid output rate | share of answers that pass schema validation (first try / after retries) | all |
| Latency | wall-clock seconds per question, p50 and p95 | all |
| Tokens | mean tokens in / out per question | LLM systems |
| Cost | € per question (API: tokens × price on TODO date; local: 0 € API cost) | all |

Retrieval is scored at **aid level**: a chunk counts as its parent `id_aid`, duplicates removed before ranking.

### Judge

<!-- TODO -->
- Judge model: TODO (exact version), prompt in `TODO path`.
- Rubric: TODO.
- Validation: we labelled TODO dev answers by hand; judge agreement with our labels = TODO %.
- Known biases: TODO (e.g. lenient on partial answers, prefers longer answers).

---

## 4. Results

### Held-out (q021–q040)

<!-- TODO: from eval/summary.json -->
| System | R@1 | R@5 | R@10 | MRR | Correct | Behaviour OK | Cit. precision | p50 / p95 latency (s) | Tokens in / out | € / question |
|---|---|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.75 | 0.88 | 0.94 | 0.81 | n/a | n/a | n/a | < 0.01 | 0 / 0 | 0 |
| `hybrid` | **0.88** | **1.00** | 1.00 | **0.92** | n/a | n/a | n/a | 0.22 (mean) | 0 / 0 | 0 |
| `hybrid_rerank` ¹ | 0.75 | 0.94 | 1.00 | 0.83 | n/a | n/a | n/a | 20.5 (mean) | 0 / 0 | 0 |
| `rag` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `agent` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `agent_api` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

¹ Measured with `bench_rerank.py` on a hybrid pool with equal weights (`eval/retrieval_rerank.json`), not on the exact `hybrid_rerank` pool (dense weight 3); latency = reranking only. Retrieval rows: 16 held-out questions with gold docs; latency = mean query time on the demo machine (CPU), from `eval/retrieval_dense.json`.

Dev scores for the same systems: `eval/summary_dev.json` <!-- TODO: confirm file name -->.

### By category (held-out, final system)

| Category | n | Correct / behaviour OK | Notes |
|---|---:|---|---|
| factual | TODO | TODO | TODO |
| eligibility | TODO | TODO | TODO |
| geographic | TODO | TODO | TODO |
| deadline | TODO | TODO | TODO |
| exclusion | TODO | TODO | TODO |
| no_amount | TODO | TODO | TODO |
| out_of_corpus | TODO | TODO | TODO |
| prompt_injection | TODO | TODO | TODO |

### Cost / latency trade-off

<!-- TODO: one plot (quality vs p50 latency, one point per system) + 2–3 sentences -->
TODO: which system we would ship under a latency budget of TODO s, and why.

---

## 5. Experiments (dev split)

One design decision per row, everything else fixed. Raw outputs in `eval/experiments/` <!-- TODO: confirm -->.

| # | Decision | Variant A | Variant B | Metric | A | B | Kept |
|---|---|---|---|---|---|---|---|
| E1 | Chunk size (body chars, BM25) | 1,800 | 1,200 / whole aid | Recall@5 / MRR | 1.00 / 0.84 | 1.00 / 0.87 ; 1.00 / 0.85 | 1,800 ² |
| E2 | Metadata header in each chunk | with | without | Recall@5 | **1.00** | 0.77 | with |
| E3 | BM25 analyzer | French stemming | no stemming | Recall@5 / MRR | 1.00 / 0.84 | 1.00 / 0.88 | stemming ³ |
| E4 | Retriever | BM25 | hybrid, bge-m3 ×3 | Recall@1 / MRR | 0.74 / 0.84 | **0.91 / 0.97** | hybrid ⁴ |
| E5 | Reranker (pool 10) | off | `bge-reranker-v2-m3` | MRR / latency | 0.94 / 0 s | 0.97 / +20.5 s | off ⁵ |
| E6 | Chunks passed to the LLM | top TODO | top TODO | correctness / tokens | TODO | TODO | TODO |
| E7 | RAG vs agent | `rag` | `agent` | correctness / latency | TODO | TODO | TODO |
| E8 | Local vs API LLM | TODO | TODO | correctness / latency / cost | TODO | TODO | TODO |

Retrieval rows are on the 17 dev questions with gold docs. Sources: `eval/retrieval_bm25.json` (E1–E3), `eval/retrieval_dense.json` and `eval/retrieval_fusion.json` (E4), `eval/retrieval_rerank.json` (E5).

² Ties on Recall@5; 1,800 characters keeps header + body inside a 512-token encoder window for 87% of chunks (390 median / 636 max real tokens, bge-m3 tokenizer). 1,200 has a slightly higher MRR, not re-tested with the dense model.
³ Honest note: dev slightly favours no stemming (MRR 0.88 vs 0.84; held-out Recall@5 1.00 vs 0.88). We kept stemming; the impact is limited because BM25 weighs 1/4 in the final fusion. Not re-tuned.
⁴ Embedding model chosen first, dense alone (Recall@1 dev / held-out): e5-small 0.74 / 0.75, e5-base 0.74 / 0.63, **bge-m3 0.91 / 0.88** (query 39 / 86 / 219 ms, RAM 1.6 / 2.4 / 2.6 GB). Fusion weight then picked on dev with a rule fixed beforehand (Recall@1, then MRR, then simplest): equal weights give 0.85 / 0.75, dense ×3 gives 0.91 / 0.88 (held-out checked afterwards).
⁵ `mmarco-mMiniLMv2` (0.5 GB) made it worse: held-out Recall@1 0.75 → 0.50 at pool 10 (0.44 at 20 and 50), +2.6 s. `bge-reranker-v2-m3`: dev Recall@1 0.85 → 0.91 but held-out unchanged (0.75), +20.5 s per question on CPU, identical when re-measured at 1,024 tokens instead of 512. Kept as an option.

---

## 6. Error analysis

<!-- TODO: read every failed held-out question of the final system; one row per question, then group -->

### Failure taxonomy (held-out, final system)

| Failure type | Count | Questions | Cause | Possible fix |
|---|---:|---|---|---|
| Retrieval miss (gold aid not in top k) | TODO | TODO | TODO | TODO |
| Right aid retrieved, wrong answer | TODO | TODO | TODO | TODO |
| Wrong or missing citation | TODO | TODO | TODO | TODO |
| Answered instead of abstaining | TODO | TODO | TODO | TODO |
| Abstained although the answer was there | TODO | TODO | TODO | TODO |
| Injection followed | TODO | TODO | TODO | TODO |
| Invented amount (`no_amount`) | TODO | TODO | TODO | TODO |
| Invalid structured output | TODO | TODO | TODO | TODO |

### Worked examples

<!-- TODO: 2–3 failures, each with question, expected, got, retrieved ids, diagnosis -->

**qXXX (TODO category)**
- Question: TODO
- Expected: TODO (gold `TODO`)
- Got: TODO
- Retrieved: TODO
- Diagnosis: TODO

---

## 7. Residual risks

| Risk | Evidence | Mitigation | Remaining |
|---|---|---|---|
| Stale data | Snapshot of 2026-10-13; official files are updated weekly; 168 aids have a deadline | Answers show the aid's `aid_validation` date | Deadlines and amounts may have changed |
| Region filters miss aids | 204 of 432 territorial aids have no machine-readable region (see `DATA_CARD.md` §6–7) | Retrieval does not hard-exclude `regions = []`; TODO | TODO |
| Dead source links | 15 of 524 URLs return 404, 122 unverified | Citations use `id_aid`; URLs are secondary | TODO |
| Hallucination | TODO: unsupported-claim and hallucinated-citation rates | Citation check against retrieved ids; TODO | TODO |
| Invented amounts | TODO: `no_amount` results | TODO | TODO |
| Prompt injection | TODO: injection results | Retrieved text treated as data; TODO | TODO |
| Not legal / financial advice | Eligibility depends on details the corpus does not hold | Answers are indicative and point to the funder | Users may over-trust answers |
| Small golden set | 20 held-out questions: one question = 5 points | Report counts, not only percentages | Differences under ~10 points are not conclusive |
| Judge bias | TODO: judge vs human agreement | TODO | TODO |
| Latency on CPU | TODO s p50 | TODO | Limits interactive use |

---

## 8. Reproduce

```bash
uv run python -m aides_rag.retrieval.eval_recall                                   # retrieval-only metrics (BM25 variants)
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --system <name>
# TODO: judge step, if separate
# TODO: script that builds the tables of §4 from eval/*.json
```

Outputs: `eval/results.jsonl` (one line per question) and `eval/summary.json` (aggregates).
