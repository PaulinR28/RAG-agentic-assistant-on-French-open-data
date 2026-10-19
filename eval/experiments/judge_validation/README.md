# Judge validation

Does the LLM judge (`src/aides_rag/evaluation/judge.py`, Claude Sonnet 5.5) grade like us?

Answers: the two dev runs of e9 (`rag`, local Qwen3 4B, 20 questions each, before / after the place hints), graded by
hand in `../e9_full_cards_geo/README.md` (copied in `labels.json`). Run: `uv run python eval/experiments/judge_validation/validate.py`.

| | Value |
|---|---|
| Answers | 40 |
| Same verdict as the hand label | **39 / 40 (97.5 %)** |
| Same correct vs not-correct decision | 39 / 40 (97.5 %) |
| Confusion (human → judge) | correct→correct 29, wrong→wrong 8, partial→partial 2, correct→partial 1 |
| Judge cost | $0.12 for 40 verdicts |

Only disagreement: q006 (after), hand `correct`, judge `partial`: "matches the reference on amount, duration, rate
and deferral, but omits 'no guarantee' and the mandatory insurance, and adds unsupported eligibility conditions".
The judge is stricter than us here, and its reason holds.

Files: `judged_before.jsonl`, `judged_after.jsonl` (verdict + reason per answer), `agreement.json`.
