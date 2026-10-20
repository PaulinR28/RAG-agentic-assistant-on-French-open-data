"""Offline check of a possible validator rule: every percentage and euro amount of an answer must be found in
the cited aid cards. Run on the final held-out answers; no LLM call, the assistant is unchanged.

    uv run python eval/experiments/numeric_grounding/check.py

Two variants:
- strict: every number must be in the cited cards;
- lenient: percentages must be in the cited cards (never taken from the question: in q040 the "100 %" is the
  injection), amounts may also come from the question (the user's own figures) or be a card percentage applied
  to one of them (q032: 30 % of 8,000 € = 2,400 €).
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("AIDES_SEARCH", "bm25")  # get_aid only: no need to load the dense model
from aides_rag.agent.tools import run_tool  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("flags.json")
SYSTEMS = ["rag", "agent", "rag_api", "agent_api"]
SP = r"[\s  ]"  # spaces, incl. the non-breaking and narrow ones of French numbers
NUM = rf"\d{{1,3}}(?:{SP}\d{{3}})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
PCT = re.compile(rf"({NUM}){SP}?%")
EUR = re.compile(rf"({NUM}){SP}?(k€|K€|M€|millions?{SP}?(?:d'euros|€)?|Millions?{SP}?€?|€|euros?)")


def to_float(s: str) -> float:
    return float(re.sub(SP, "", s).replace(",", "."))


def pcts(text: str) -> set[float]:
    return {to_float(n) for n in PCT.findall(text)}


def amounts(text: str) -> set[float]:
    out = set()
    for n, unit in EUR.findall(text):
        scale = 1e3 if unit.lower().startswith("k") else 1e6 if unit.lower().startswith("m") else 1
        out.add(round(to_float(n) * scale, 2))
    return out


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main() -> None:
    golden = load(ROOT / "eval/golden.jsonl")
    cards: dict[str, str] = {}
    rows = []
    for system in SYSTEMS:
        results = load(ROOT / f"eval/final/{system}/results.jsonl")
        judged = load(ROOT / f"eval/final/{system}/judged.jsonl")
        for qid, r in results.items():
            if r["abstained"] or not r["answer"]:
                continue
            for aid in r["cited_sources"] or []:
                cards.setdefault(aid, run_tool("get_aid", {"id_aid": aid}, set())[0])
            src = "\n".join(cards[a] for a in r["cited_sources"] or [])
            question = golden[qid]["question"]
            ans_p, ans_a = pcts(r["answer"]), amounts(r["answer"])
            src_p = pcts(src)
            # card side: any number may be an amount ("entre 50 000 et 5 000 000 €": € only after the last one)
            src_a = amounts(src) | {round(to_float(n), 2) for n in re.findall(NUM, src)}
            q_a = amounts(question)
            derived = {round(p / 100 * a, 2) for p in src_p for a in q_a}
            rows.append({"system": system, "id": qid, "grade": judged[qid]["correctness"],
                         "strict": sorted(ans_p - src_p) + sorted(ans_a - src_a),
                         "lenient": sorted(ans_p - src_p) + sorted(ans_a - src_a - q_a - derived)})

    totals = Counter(r["grade"] for r in rows)
    summary = {"answers_checked": len(rows), "by_grade": dict(totals)}
    for mode in ("strict", "lenient"):
        flagged = [r for r in rows if r[mode]]
        hit = Counter(r["grade"] for r in flagged)
        summary[mode] = {"flagged": len(flagged), "wrong": hit["wrong"], "partial": hit["partial"],
                         "correct_false_alarms": hit["correct"],
                         "rows": [{k: r[k] for k in ("system", "id", "grade")} | {"unsupported": r[mode]}
                                  for r in flagged]}
        print(f"{mode:7} flagged {len(flagged):2} | wrong {hit['wrong']}/{totals['wrong']}  "
              f"partial {hit['partial']}/{totals['partial']}  correct (false alarms) {hit['correct']}/{totals['correct']}")
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
