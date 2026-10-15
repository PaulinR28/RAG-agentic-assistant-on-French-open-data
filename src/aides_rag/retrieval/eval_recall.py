"""Retrieval-only evaluation: Recall@k and MRR at aid level on the golden questions that have gold_docs.

Chunk scores are aggregated per aid (max), so an aid counts as retrieved if any of its chunks is.
Recall@k = share of a question's gold aids found in the top k aids, averaged over questions.
Compares a few chunking / analyzer variants so each choice is backed by a number.

Usage:
    uv run python -m aides_rag.retrieval.eval_recall --golden eval/golden.jsonl --out eval/retrieval_bm25.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aides_rag.index.chunks import MAX_BODY_CHARS, build_chunks, read_corpus
from aides_rag.retrieval.bm25 import BM25

KS = (1, 5, 10, 50)

VARIANTS = {
    "bm25_stem_header": dict(stem=True, header=True, budget=MAX_BODY_CHARS),
    "bm25_nostem_header": dict(stem=False, header=True, budget=MAX_BODY_CHARS),
    "bm25_stem_noheader": dict(stem=True, header=False, budget=MAX_BODY_CHARS),
    "bm25_stem_header_1200": dict(stem=True, header=True, budget=1200),
    "bm25_stem_header_wholeaid": dict(stem=True, header=True, budget=100_000),
}
DEFAULT = "bm25_stem_header"


def rank_aids(index: BM25, chunk_aids: np.ndarray, query: str) -> list[str]:
    """Aids ordered by their best chunk score (aids with score 0 are left out)."""
    scores = index.scores(query)
    best: dict[str, float] = {}
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] <= 0:
            break
        best.setdefault(chunk_aids[i], float(scores[i]))
    return list(best)


def evaluate(aids: list[dict], questions: list[dict], stem: bool, header: bool, budget: int) -> dict:
    chunks = build_chunks(aids, budget)
    index = BM25([c["text"] if header else c["body"] for c in chunks], stem=stem)
    chunk_aids = np.array([c["id_aid"] for c in chunks])
    per_question = []
    for q in questions:
        ranking = rank_aids(index, chunk_aids, q["question"])
        ranks = {g: ranking.index(g) + 1 if g in ranking else None for g in q["gold_docs"]}
        per_question.append({"id": q["id"], "category": q["category"], "gold_ranks": ranks, "top5": ranking[:5]})
    summary = {f"recall@{k}": round(float(np.mean([
        np.mean([r is not None and r <= k for r in p["gold_ranks"].values()]) for p in per_question])), 3)
        for k in KS}
    summary["mrr"] = round(float(np.mean([
        1 / min(r for r in p["gold_ranks"].values() if r) if any(p["gold_ranks"].values()) else 0.0
        for p in per_question])), 3)
    summary["n_chunks"] = len(chunks)
    return {"summary": summary, "per_question": per_question}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/aids.jsonl"))
    ap.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/retrieval_bm25.json"))
    args = ap.parse_args()

    aids = read_corpus(args.corpus)
    questions = [q for q in read_corpus(args.golden) if q["gold_docs"]]
    results = {name: evaluate(aids, questions, **cfg) for name, cfg in VARIANTS.items()}

    print(f"{len(questions)} questions with gold docs\n")
    print(f"{'variant':<28}" + "".join(f"{m:>11}" for m in [*(f'recall@{k}' for k in KS), "mrr", "n_chunks"]))
    for name, res in results.items():
        print(f"{name:<28}" + "".join(f"{v:>11}" for v in res["summary"].values()))
    print(f"\nmisses at k=5 ({DEFAULT}):")
    for p in results[DEFAULT]["per_question"]:
        if not all(r and r <= 5 for r in p["gold_ranks"].values()):
            print(f"  {p['id']} {p['category']:<12} gold ranks {p['gold_ranks']}")

    out = {
        "questions": len(questions),
        "default": DEFAULT,
        "variants": {name: {"config": VARIANTS[name], **res} for name, res in results.items()},
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
