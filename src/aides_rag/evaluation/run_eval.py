"""Run a system on the golden questions and score it.

Writes, in --out:
  results.jsonl  one line per question: answer, cited sources, retrieved aid ids, tokens in/out,
                 latency, tool calls, stop reason (+ id, category, gold_docs, error)
  summary.json   retrieval metrics (Recall@k, MRR; all questions and per split dev / heldout),
                 latency, tokens, tool calls, errors

Retrieval metrics are aid-level and use the questions that have gold_docs, as in
retrieval/eval_recall.py. Answer-level scores (judged correctness, abstention, injection) are
reported as null until a judge exists.

Errors count both exceptions caught here and errors a system reports itself (the generating
systems catch their own exceptions and return stop_reason "error"), so a broken run never looks clean.

Usage:
    uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ [--system rag]

The default system is the assistant itself (rag), so the brief's command scores the answers, not
retrieval alone. Retrieval-only systems: --system bm25 | hybrid | hybrid_rerank.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from aides_rag.index.chunks import read_corpus
from aides_rag.systems import OUTPUT_KEYS, SYSTEMS, load_system

KS = (1, 5, 10)
SPLITS = ("dev", "heldout")
DEFAULT_SYSTEM = "rag"


def run(system, questions: list[dict], system_name: str) -> list[dict]:
    rows = []
    for q in questions:
        start = time.perf_counter()
        try:
            out, error = system.answer(q["question"]), None
        except Exception as exc:  # one failing question must not stop the run
            out, error = {}, f"{type(exc).__name__}: {exc}"
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        rows.append({
            "id": q["id"],
            "system": system_name,
            "split": q.get("split"),
            "category": q.get("category"),
            "expected_behavior": q.get("expected_behavior"),
            "question": q["question"],
            "gold_docs": q["gold_docs"],
            **{key: out.get(key) for key in OUTPUT_KEYS},
            "latency_ms": latency_ms,
            "error": error or out.get("error"),  # systems may report their own errors
        })
    return rows


def gold_ranks(row: dict) -> list[int | None]:
    retrieved = row["retrieved_ids"] or []
    return [retrieved.index(g) + 1 if g in retrieved else None for g in row["gold_docs"]]


def retrieval_metrics(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["gold_docs"]]
    if not scored:
        return {"n_questions": 0}
    ranks = [gold_ranks(r) for r in scored]
    metrics = {"n_questions": len(scored)}
    for k in KS:
        metrics[f"recall@{k}"] = round(float(np.mean([np.mean([x is not None and x <= k for x in rk]) for rk in ranks])), 3)
    metrics["mrr"] = round(float(np.mean([1 / min(x for x in rk if x) if any(rk) else 0.0 for rk in ranks])), 3)
    by_category = {}
    for category in sorted({r["category"] for r in scored}):
        cat = [rk for r, rk in zip(scored, ranks) if r["category"] == category]
        by_category[category] = round(float(np.mean([np.mean([x is not None and x <= 5 for x in rk]) for rk in cat])), 3)
    metrics["recall@5_by_category"] = by_category
    metrics["misses@5"] = [r["id"] for r, rk in zip(scored, ranks) if not all(x and x <= 5 for x in rk)]
    return metrics


def total(rows: list[dict], key: str) -> int | None:
    values = [r[key] for r in rows if r[key] is not None]
    return sum(values) if values else None


def summarize(rows: list[dict], system_name: str) -> dict:
    latencies = np.array([r["latency_ms"] for r in rows])
    return {
        "system": system_name,
        "n_questions": len(rows),
        "errors": [r["id"] for r in rows if r["error"] or r["stop_reason"] == "error"],
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows if r["stop_reason"])),
        "retrieval": retrieval_metrics(rows),
        # tune on dev, report final scores on heldout (no aid is shared between the two splits)
        "retrieval_by_split": {split: retrieval_metrics([r for r in rows if r["split"] == split])
                               for split in SPLITS if any(r["split"] == split for r in rows)},
        "latency_ms": {
            "mean": round(float(latencies.mean()), 1),
            "p50": round(float(np.percentile(latencies, 50)), 1),
            "p95": round(float(np.percentile(latencies, 95)), 1),
            "max": round(float(latencies.max()), 1),
        },
        "tokens": {"in": total(rows, "tokens_in"), "out": total(rows, "tokens_out")},
        "cost_usd": round(cost, 6) if (cost := total(rows, "cost_usd")) is not None else None,  # billed, 0 when local
        "models": sorted({r["model"] for r in rows if r["model"]}),
        "configs": sorted({r["config"] for r in rows if r["config"]}),
        "tool_calls": sum(len(r["tool_calls"] or []) for r in rows),
        "answers_generated": sum(bool(r["answer"]) for r in rows),  # an empty "" is not an answer
        # filled in once a judge exists
        "judged_correctness": None,
        "expected_behavior_respected": None,
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval"))
    ap.add_argument("--system", default=DEFAULT_SYSTEM, choices=sorted(SYSTEMS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (quick check)")
    args = ap.parse_args(argv)

    questions = read_corpus(args.questions)[: args.limit]
    system = load_system(args.system)
    rows = run(system, questions, args.system)
    summary = summarize(rows, args.system)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.out / "results.jsonl")
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"system={args.system}  questions={len(rows)}  errors={len(summary['errors'])}")
    for name, ret in [("all", summary["retrieval"]), *summary["retrieval_by_split"].items()]:
        if ret["n_questions"]:
            print(f"  [{name}] " + "  ".join(f"{m}={ret[m]}" for m in [*(f"recall@{k}" for k in KS), "mrr"])
                  + f"  (on {ret['n_questions']} questions with gold docs)")
    lat = summary["latency_ms"]
    print(f"  latency ms: mean={lat['mean']} p95={lat['p95']}  tokens: {summary['tokens']}  tool calls: {summary['tool_calls']}")
    print(f"-> {args.out / 'results.jsonl'}, {args.out / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
