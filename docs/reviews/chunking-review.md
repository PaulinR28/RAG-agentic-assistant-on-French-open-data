# Chunking review: 20 random aids + 20 random chunks

**Status: ✅ CONFIRMED** (2026-10-16). The chunking strategy is kept for the dense index.

Follows the brief's tip: *"open 20 random documents and 20 random chunks before choosing a chunking strategy"*.

## Sample (reproducible)

```python
import json, random
aids = [json.loads(l) for l in open("data/processed/aids.jsonl", encoding="utf-8")]
chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
rng = random.Random(42)
sample_aids, sample_chunks = rng.sample(aids, 20), rng.sample(chunks, 20)
```

- Aids: 12071, 8629, 6876, 12287, 10796, 10576, 10355, 8923, 12263, 8364, 12127, 12644, 11722, 8233, 11914, 11472, 7473, 7141, 8295, 10321
- Chunks: 10672-0, 11881-2, 12149-1, 6880-0, 12091-2, 10334-0, 12476-0, 12303-0, 12444-0, 12047-1, 11565-0, 10574-1, 11627-1, 12131-0, 10984-0, 12774-1, 411-4, 12630-1, 12766-0, 9753-1

## Strategy under review

One aid = one chunk. Aids longer than 1,800 body characters are split by section (objet, operations,
conditions, montant, beneficiaires), with ~15% overlap inside long sections. Every chunk starts with a
metadata header (title, aid type, area, funders, themes, deadline, update date). `%pme%` and `%mini%`
are expanded.

## What we saw

| # | Observation | Verdict |
|---|---|---|
| 1 | The answers to typical questions (rate, ceiling, who is eligible) sit in `montant` and `beneficiaires`, which are short and kept whole | ✅ Section split fits the data |
| 2 | A chunk holding only `montant` (e.g. 11881-2) is still understandable: the header gives aid name, area, type | ✅ The header is needed (Recall@5 without it: 0.765, with it: 1.0) |
| 3 | 18 of 20 sampled chunks are 300–610 tokens; the smallest is 279 | ✅ Inside the 300–800 target |
| 4 | Placeholders appear in real text (`%mini%` at the end of `montant`, `%pme%` alone in `beneficiaires`) | ✅ Expanded correctly |
| 5 | Split sections lost their line breaks (lists became one dense block) | 🔧 **Fixed in this review**: line breaks preserved; Recall unchanged |
| 6 | Some sections are long lists without punctuation (12149-1, 411-4): a split can fall mid-sentence | ⚠️ Accepted: rare, and the overlap repeats the context in the next chunk |
| 7 | Many aids list **exclusions** ("Ne sont pas éligibles : pompes à chaleur…", 12071, 12774) | ⚠️ Retrieval will match the excluded term. **The LLM prompt must respect negations**; add one golden question on this |
| 8 | Some amounts are not stated ("montant dépendra du projet", 12131, 12766) | ⚠️ The LLM must say so rather than invent a figure |
| 9 | A few in-theme aids are only loosely green or innovative (tourism lodging, press fund, shop fronts) | ℹ️ Corpus scope, not chunking: they are tagged with the theme in the source. Noted for the data card |
| 10 | Local aids without a resolved region (8629 Nevers, 10576 Val Marnaysien) still show their area name in the header | ✅ Searchable by place name |

## Decision

- **Kept:** 1 aid = 1 chunk, section split for long aids, metadata header, placeholder expansion.
- **Changed:** line breaks preserved when splitting.
- **Carried to later steps:** a negation rule in the LLM prompt, a "not stated in the document" rule
  for amounts, and 2 held-out golden questions (an excluded expense, an unstated amount).
