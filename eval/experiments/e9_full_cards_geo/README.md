# E9: whole cards + place hints (dev, rag, local qwen3 4B)

Setup: `run_eval --system rag` on the 20 **dev** questions, `qwen3:4b-instruct-2507-q4_K_M` via Ollama (Intel iGPU,
Vulkan), hybrid search, k = 5. **Before**: best chunk per aid, no place hints (`AIDES_FULL_CARDS=0 AIDES_GEO=0`).
**After**: whole cards (aids ≤ 4,000 characters) + place hints. Grading: manual, against `answer` in
`eval/golden.jsonl`, by Izan (correct / partial / wrong).

| Question | Before | After | Note |
|---|---|---|---|
| q005 (12 t electric truck) | correct | **wrong** | whole card: model reads the retrofit paragraph as "thermal vehicles only" |
| q007 (company 2 years old) | wrong | wrong | "Oui… ne remplit pas la condition": self-contradiction |
| q010 (baker created in 2025) | wrong | wrong | before: condition not shown (chunk 3/3). After: shown, but "2025 respects created before 2023" |
| q011 (cargo bike in Lyon) | wrong | **correct** | place hint "Lyon → Auvergne-Rhône-Alpes" |
| q012 (Bretagne loan) | partial | partial | rate given as "0,05 %" instead of "TMO – 0,05 %" |
| q016 (deadline passed) | wrong | wrong | "Oui… cependant la date limite est dépassée" |
| other 14 | correct | correct | |
| **Total** | **15 / 1 / 4** | **15 / 1 / 4** | |

Prompt tokens: 62,829 → 77,074 (**+23%**). Mean latency: 61.7 s → 64.1 s.

Decision: **place hints ON** (fixes q011, no regression). **Whole cards OFF by default** (no gain, one regression,
+23% tokens with a 4B model); kept as a per-question option (`agent.answer(..., full_cards=True)`, UI toggle).
The remaining errors are reasoning errors of the 4B model (self-contradiction, date comparison), not retrieval:
the gold aid is cited in all of them.
