"""Judge validation: agreement between the LLM judge and our hand labels on 40 dev answers.

    uv run python eval/experiments/judge_validation/validate.py

The answers are the two dev runs of experiment e9 (rag, local Qwen3 4B, before / after the place hints),
graded by hand in eval/experiments/e9_full_cards_geo/README.md; the labels are copied in labels.json.
Writes judged_before.jsonl, judged_after.jsonl and agreement.json next to this file.
"""

import json
from collections import Counter
from pathlib import Path

from aides_rag.evaluation.judge import judge_run
from aides_rag.index.chunks import read_corpus

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E9 = ROOT / "eval/experiments/e9_full_cards_geo"


def main() -> dict:
    labels = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))
    golden = read_corpus(ROOT / "eval/golden.jsonl")
    pairs, report = [], {}
    for run in ("before", "after"):
        judged = judge_run(read_corpus(E9 / f"{run}_results.jsonl"), golden)
        (HERE / f"judged_{run}.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in judged),
                                                  encoding="utf-8", newline="\n")
        for j in judged:
            pairs.append((run, j["id"], labels[run][j["id"]], j["correctness"], j["reason"]))
    agree = [p for p in pairs if p[2] == p[3]]
    # the decision that matters most: is the answer usable (correct) or not (partial / wrong)?
    agree_binary = [p for p in pairs if (p[2] == "correct") == (p[3] == "correct")]
    report = {
        "n": len(pairs),
        "exact_agreement": round(len(agree) / len(pairs), 3),
        "correct_vs_not_agreement": round(len(agree_binary) / len(pairs), 3),
        "confusion_human_to_judge": {f"{h}->{m}": c for (h, m), c in sorted(Counter((p[2], p[3]) for p in pairs).items())},
        "disagreements": [{"run": r, "id": i, "human": h, "judge": m, "judge_reason": why}
                          for r, i, h, m, why in pairs if h != m],
    }
    (HERE / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                                         newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=2))
    for d in report["disagreements"]:
        print(f"  {d['run']} {d['id']}: human={d['human']} judge={d['judge']}  ({d['judge_reason']})")
    return report


if __name__ == "__main__":
    main()
