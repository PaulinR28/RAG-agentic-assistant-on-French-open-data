# Evaluation

How we know the assistant works, how well, at what cost, and where it fails.

Course doctrine: no component counts as working until it is scored on the held-out golden set with
cost and latency tracked. Every design decision below was tested on the **dev** split; the **held-out**
split was run once per final system, for reporting only.

Final runs: 2026-10-19, commit `ca93aac` (assistant frozen), MacBook Air M3 16 GB, Ollama 0.40.6 (`eval/final/RUN_INFO.txt`).
Held-out was run once per system, after the final system (`rag`, local) had been declared in PR #20. Nothing was tuned after it.

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
- How they were written: all 40 were drafted with an AI assistant from the aid cards, then read and approved by the
  team (dev on 2026-10-14; held-out reviewed before merge in PR #5). Because they were written from the cards, they
  may share more words with them than real user questions: BM25 alone already ranks the gold aid first on 12 of
  the 16 held-out questions with gold docs.

**Composition**

| Category | What it tests | dev | heldout |
|---|---|---:|---:|
| factual | a fact stated in one aid (amount, funder, rate) | 6 | 4 |
| eligibility | can company X get aid Y (size, age, sector) | 4 | 3 |
| geographic | aids available in a given region | 3 | 3 |
| deadline | dates, "still open on <date>" | 3 | 2 |
| exclusion | conditions that rule a company out | 0 | 2 |
| no_amount | aid exists but the corpus gives no amount: must not invent one | 0 | 1 |
| out_of_corpus | no answer in the corpus: must abstain | 2 | 3 |
| prompt_injection | injected instruction in the question: refuse or ignore it | 2 | 2 |
| **Total** | | **20** | **20** |

---

## 2. Systems compared

All systems implement the same `answer(question) -> dict` contract (`src/aides_rag/systems/__init__.py`)
and are run with:

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --system <name>
```

| System | Retrieval | Generation | Notes |
|---|---|---|---|
| `bm25` | BM25, French analyzer | none | baseline |
| `hybrid` | BM25 + `bge-m3`, weighted RRF (k = 60, dense weight 3) | none | retrieval of `rag` / `agent` |
| `hybrid_rerank` | hybrid, top 10 chunks re-sorted by `bge-reranker-v2-m3` | none | option, off by default |
| `rag` | one forced `hybrid` search with the raw question, top 5 aids | `qwen3:4b-instruct-2507-q4_K_M` (Ollama), one answer call (+1 retry if invalid) | baseline with an LLM |
| `agent` | tools `search_aids` (`hybrid`, filters: region, aid type, deadline status on a date) and `get_aid` (full card) | same model, up to 3 tool calls, then one answer call (+1 retry) | |

API comparison: the same `rag` or `agent` system with `LLM_BACKEND=openrouter LLM_MODEL=<model>` (Claude Haiku 5.5,
Qwen3-14B). There is no separate API system. Final system (`rag` or `agent`): TODO, decided on dev.

---

## 3. Metrics

| Metric | Definition | Computed on |
|---|---|---|
| Recall@k (k = 1, 5, 10) | share of questions with at least one gold doc in the top k retrieved aids | questions with non-empty `gold_docs` |
| MRR | mean of 1 / rank of the first gold doc (0 if absent from the top 10) | same |
| Correctness | judged `correct` / `partial` / `wrong` against the reference answer (LLM judge, below); accuracy = share `correct`. For an expected abstention or refusal, `correct` = declined clearly | all questions |
| Citation precision | share of cited ids that are gold docs or support the answer | answers with citations |
| Hallucinated citations | cited ids that were not retrieved: rejected by the validator (retry, then `invalid_output`), so 0 in valid answers by construction | all answers |
| Behaviour accuracy | abstains, refuses or ignores the injection as expected | `abstain`, `refuse_injection`, `answer_ignore_injection` |
| Valid output rate | share of answers that pass schema validation (first try / after the one retry) | all |
| Latency | wall-clock seconds per question, p50 and p95 | all |
| Tokens | mean tokens in / out per question | LLM systems |
| Cost | USD per question: the cost billed by OpenRouter (`usage.cost`); 0 for the local model | all |

Retrieval is scored at **aid level**: a chunk counts as its parent `id_aid`, duplicates removed before ranking.

### Judge

- **Judge model:** `anthropic/claude-sonnet-5.5` via OpenRouter, temperature 0, reasoning effort `low` (the model
  cannot run without reasoning). It is a different, larger model than the graded ones (Qwen3 4B, Claude Haiku 5.5).
  Prompt and rubric: `src/aides_rag/evaluation/judge.py`. Run it with `run_eval.py --judge`, or afterwards on a
  results file with `python -m aides_rag.evaluation.judge`. Output: `eval/judged.jsonl` (verdict and one-sentence
  reason per question) and the `judged_correctness` / `expected_behavior_respected` fields of `summary.json`,
  overall and per split.
- **Rubric:** the judge compares with the reference answer only (not its own knowledge). `correct` = the key facts
  the question asks for, no contradiction; `partial` = right direction, a key element missing or imprecise;
  `wrong` = contradiction, self-contradiction, invented fact, or wrong behaviour (answers when it should abstain,
  abstains when it should answer). A row with no answer (error) is `wrong` without a judge call.
- **Validation:** the 20 **dev** questions only (held-out never shown to the judge), asked twice (the two e9
  runs, before / after the place hints) = 40 answers graded by hand (rag, local Qwen3 4B; labels in
  `eval/experiments/judge_validation/labels.json`). Judge vs hand labels: **39 / 40 identical (97.5 %)**, same
  for correct vs not-correct. The one disagreement (q006, after) is stricter than us: `partial` where we said
  `correct`, because the answer adds eligibility conditions that are not in the aid card. Cost of the 40 verdicts:
  $0.12. Details: `eval/experiments/judge_validation/agreement.json`.
- **Known limits:** only 3 `partial` answers in the validation set, so the partial / correct boundary is the least
  tested; the judge trusts the reference answer, so an error in the golden set is not caught; it may be slightly
  stricter than us on extra, unsupported details.

---

## 4. Results

### Held-out (q021–q040)

| System | R@1 | R@5 | R@10 | MRR | Correct | Behaviour OK | Cit. precision | p50 / p95 latency (s) | Tokens in / out | USD / question |
|---|---|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.75 | 0.88 | 0.94 | 0.81 | n/a | n/a | n/a | < 0.01 | 0 / 0 | 0 |
| `hybrid` | **0.88** | **1.00** | 1.00 | **0.92** | n/a | n/a | n/a | 0.22 (mean) | 0 / 0 | 0 |
| `hybrid_rerank` ¹ | 0.75 | 0.94 | 1.00 | 0.83 | n/a | n/a | n/a | 20.5 (mean) | 0 / 0 | 0 |
| **`rag`, Qwen3 4B local (final)** | 0.88 | 1.00 | 1.00 | 0.92 | **11 / 3 / 6** | 17 / 20 | 13 / 18 | 23.1 / 32.6 | 3,073 / 149 | 0 |
| `agent`, Qwen3 4B local | 0.75 | 0.88 | 0.88 | 0.78 | 11 / 3 / 6 | 19 / 20 | 13 / 21 | 26.8 / 40.3 | 9,652 / 250 | 0 |
| `rag`, Claude Haiku 5.5 (API) | 0.88 | 1.00 | 1.00 | 0.92 | 15 / 2 / 3 | 19 / 20 | 15 / 18 | 2.1 / 5.0 | 5,403 / 244 | 0.00066 |
| `agent`, Claude Haiku 5.5 (API) | 0.81 | 0.94 | 0.94 | 0.88 | **16 / 2 / 2** | 19 / 20 | 14 / 16 | 4.7 / 8.9 | 14,323 / 480 | 0.00167 |

¹ Measured with `bench_rerank.py` on a hybrid pool with equal weights (`eval/retrieval_rerank.json`), not on the exact `hybrid_rerank` pool (dense weight 3); latency = reranking only. Retrieval rows: 16 held-out questions with gold docs; latency = mean query time on the demo machine (CPU), from `eval/retrieval_dense.json`.

LLM rows: `eval/final/<system>/` (`results.jsonl`, `summary.json`, `judged.jsonl`). Correct = correct / partial / wrong by the judge (§3), all 20 questions. Behaviour OK = expected behaviour respected (answer, abstain, refuse, ignore the injection), all 20. Cit. precision = cited ids that are gold docs / all cited ids, a strict lower bound (a non-gold citation can still support the answer). For `agent`, retrieval is scored on the aids its searches returned. Tokens = mean per question; agent makes 3–5 LLM calls per question, rag 1–2. 0 errors and 20 / 20 valid structured outputs in all four runs (retries: rag 1, agent 3, rag API 3, agent API 3). Judge cost for the 80 verdicts: $0.26.

Dev scores for the same systems: `eval/summary_dev.json` <!-- TODO: confirm file name -->.

### By category (held-out, final system)

| Category | n | Correct / partial / wrong · behaviour OK | Notes |
|---|---:|---|---|
| factual | 4 | 2 / 1 / 1 · 3 | q024: cites the neighbouring campsite aid (12066) although 8119 was ranked first |
| eligibility | 3 | 0 / 1 / 2 · 3 | q025 and q026: the right aid, but a threshold misread (age, turnover); q026 is wrong for all four systems |
| geographic | 3 | 3 / 0 / 0 · 3 | |
| deadline | 2 | 1 / 0 / 1 · 2 | q032: says "still open" for an aid closed the day before (only API `agent` gets it) |
| exclusion | 2 | 1 / 0 / 1 · 1 | q034: misses that Grand Périgueux is excluded, cites a neighbouring aid |
| no_amount | 1 | 0 / 1 / 0 · 1 | |
| out_of_corpus | 3 | 3 / 0 / 0 · 3 | abstained on all three |
| prompt_injection | 2 | 1 / 0 / 1 · 1 | q039 refused; q040: ignores the injection but abstains instead of answering (40 %) |

In every wrong answer of the final system the gold aid was in the top 5: the errors are in reading, not in retrieval.

### Cost / latency trade-off

Local `rag` is free, keeps the data on the machine and answers in ~23 s (p50); it gets 11 / 20 correct. The same pipeline with Claude Haiku 5.5 gets 15 / 20 in 2 s for $0.0007 per question; `agent` with Haiku gets 16 / 20 in 5 s for $0.0017. Local `agent` adds three times the tokens and 4 s for no gain. Retrieval is identical across backends, so the gap is the answering model: a 4B model misreads thresholds and dates that a larger model gets right. Under a 10 s budget with a paid API allowed, we would ship `agent` with Haiku; without an API (privacy, cost), local `rag`. 20 questions: differences under ~10 points (2 questions) are not conclusive.

---

## 5. Experiments (dev split)

One design decision per row, everything else fixed. Raw outputs of generation experiments in `eval/experiments/`.

| # | Decision | Variant A | Variant B | Metric | A | B | Kept |
|---|---|---|---|---|---|---|---|
| E1 | Chunk size (body chars, BM25) | 1,800 | 1,200 / whole aid | Recall@5 / MRR | 1.00 / 0.84 | 1.00 / 0.87 ; 1.00 / 0.85 | 1,800 ² |
| E2 | Metadata header in each chunk | with | without | Recall@5 | **1.00** | 0.77 | with |
| E3 | BM25 analyzer | French stemming | no stemming | Recall@5 / MRR | 1.00 / 0.84 | 1.00 / 0.88 | stemming ³ |
| E4 | Retriever | BM25 | hybrid, bge-m3 ×3 | Recall@1 / MRR | 0.74 / 0.84 | **0.91 / 0.97** | hybrid ⁴ |
| E5 | Reranker (pool 10) | off | `bge-reranker-v2-m3` | MRR / latency | 0.94 / 0 s | 0.97 / +20.5 s | off ⁵ |
| E6 | Chunks passed to the LLM | top TODO | top TODO | correctness / tokens | TODO | TODO | TODO |
| E7 | RAG vs agent (local 4B) ⁶ | `rag` | `agent` | correct / partial / wrong, mean latency | 15 / 3 / 2, 22.7 s | 15 / 2 / 3, 37.2 s | `rag` |
| E8 | Local vs API LLM (`rag`) ⁶ | Qwen3 4B, local | Claude Haiku 5.5, API | correct / partial / wrong, mean latency, cost for 20 q | 15 / 3 / 2, 22.7 s, $0 | 19 / 0 / 1, 2.1 s, $0.011 | local (brief requires it); API as option |
| E9 | Whole cards + place hints (`rag`, local 4B) ⁷ | best chunk, no hints | whole cards (≤ 4,000 chars) + place hints | correct / partial / wrong, prompt tokens | 15 / 1 / 4, 62.8k | 15 / 1 / 4, 77.1k | hints on, whole cards off |

Retrieval rows are on the 17 dev questions with gold docs. Sources: `eval/retrieval_bm25.json` (E1–E3), `eval/retrieval_dense.json` and `eval/retrieval_fusion.json` (E4), `eval/retrieval_rerank.json` (E5). Generation rows (E7–E9) are on the 20 dev questions, graded by hand by one team member.

² Ties on Recall@5; 1,800 characters keeps header + body inside a 512-token encoder window for 87% of chunks (390 median / 636 max real tokens, bge-m3 tokenizer). 1,200 has a slightly higher MRR, not re-tested with the dense model.
³ Honest note: dev slightly favours no stemming (MRR 0.88 vs 0.84; held-out Recall@5 1.00 vs 0.88). We kept stemming; the impact is limited because BM25 weighs 1/4 in the final fusion. Not re-tuned.
⁴ Embedding model chosen first, dense alone (Recall@1 dev / held-out): e5-small 0.74 / 0.75, e5-base 0.74 / 0.63, **bge-m3 0.91 / 0.88** (query 39 / 86 / 219 ms, RAM 1.6 / 2.4 / 2.6 GB). Fusion weight then picked on dev with a rule fixed beforehand (Recall@1, then MRR, then simplest): equal weights give 0.85 / 0.75, dense ×3 gives 0.91 / 0.88 (held-out checked afterwards).
⁵ `mmarco-mMiniLMv2` (0.5 GB) made it worse: held-out Recall@1 0.75 → 0.50 at pool 10 (0.44 at 20 and 50), +2.6 s. `bge-reranker-v2-m3`: dev Recall@1 0.85 → 0.91 but held-out unchanged (0.75), +20.5 s per question on CPU, identical when re-measured at 1,024 tokens instead of 512. Kept as an option.
⁶ PR #10 (2026-10-16), measured on an 8 GB M1 before the place hints and the context-guard fix; Qwen3-14B (API): 17 / 0 / 3 (`rag`), 18 / 0 / 2 (`agent`). Final configuration: held-out table in §4.
⁷ PR #16, `eval/experiments/e9_full_cards_geo/`. Place hints fixed q011 (Lyon → Auvergne-Rhône-Alpes); whole cards broke q005 and did not fix q010 (the model still says a company created in 2025 meets "created before 30 Sept 2023").

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
