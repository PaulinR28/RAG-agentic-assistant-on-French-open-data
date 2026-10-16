"""Cheap ways to fix the hybrid ranking, no extra model: weighted RRF and re-sorting the hybrid pool by dense score.

Variants (aid level, BM25 + dense chunk scores computed once per question):
  rrf_w{w}        weighted RRF, dense weight w, BM25 weight 1 (w = 1 is the current hybrid)
  dense_rerank@N  hybrid's top N aids, re-sorted by their best dense (cosine) score; the rest keep hybrid order
The winner is picked on dev only (Recall@1, then MRR, then simplest); held-out is reported as a check.
Results go to eval/retrieval_fusion.json.

Usage:
    uv run python -m aides_rag.retrieval.bench_fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aides_rag.index.chunks import read_corpus
from aides_rag.retrieval.bm25 import BM25
from aides_rag.retrieval.dense import DEFAULT_DENSE, MODELS, DenseIndex
from aides_rag.retrieval.eval_recall import SPLITS, score_rankings
from aides_rag.retrieval.hybrid import rank_aids, rrf

WEIGHTS = (1.0, 1.5, 2.0, 3.0, 4.0)
POOLS = (5, 10, 20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", choices=sorted(MODELS), default=DEFAULT_DENSE)
    ap.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    ap.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/retrieval_fusion.json"))
    args = ap.parse_args()

    chunks = read_corpus(args.chunks)
    texts = [c["text"] for c in chunks]
    chunk_aids = np.array([c["id_aid"] for c in chunks])
    golden = read_corpus(args.golden)
    bm25 = BM25(texts)
    dense = DenseIndex(args.dense, texts)

    per_q = {}
    for q in golden:
        d = dense.scores(q["question"])
        best: dict[str, float] = {}
        for i in np.argsort(-d, kind="stable"):
            best.setdefault(chunk_aids[i], float(d[i]))
        per_q[q["question"]] = (rank_aids(bm25.scores(q["question"]), chunk_aids, positive_only=True),
                                rank_aids(d, chunk_aids), best)

    variants = {}
    for w in WEIGHTS:
        variants[f"rrf_w{w:g}"] = {question: rrf([b, dn], weights=[1.0, w]) for question, (b, dn, _) in per_q.items()}
    for n in POOLS:
        ranked = {}
        for question, (b, dn, best) in per_q.items():
            fused = rrf([b, dn])
            ranked[question] = sorted(fused[:n], key=lambda aid: -best[aid]) + fused[n:]
        variants[f"dense_rerank@{n}"] = ranked
    variants["dense_only"] = {question: dn for question, (_, dn, _) in per_q.items()}

    quality = {}
    for split in SPLITS:
        qs = [q for q in golden if q["split"] == split and q["gold_docs"]]
        quality[split] = {name: score_rankings(qs, lambda question, r=r: r[question]) for name, r in variants.items()}

    # pick on dev only: Recall@1, then MRR; ties keep the earlier (simpler) variant. dense_only is a reference, not hybrid.
    candidates = [n for n in variants if n != "dense_only"]
    pick = max(candidates, key=lambda n: (quality["dev"][n]["summary"]["recall@1"], quality["dev"][n]["summary"]["mrr"],
                                          -candidates.index(n)))

    out = {"dense": MODELS[args.dense].name, "picked_on_dev": pick, "quality": quality}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for split in SPLITS:
        print(f"[{split}]")
        for name in variants:
            s = quality[split][name]["summary"]
            mark = "  <- picked on dev" if name == pick else ""
            print(f"  {name:<16} R@1={s['recall@1']:<6} R@5={s['recall@5']:<6} MRR={s['mrr']:<6}{mark}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
