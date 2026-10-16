"""Benchmark a reranker on top of hybrid search: quality gain vs CPU cost, for several candidate pool sizes.

Pipeline per question: BM25 + dense chunk rankings -> RRF -> top POOL chunks -> cross-encoder -> aid ranking.
Reports, per split: Recall@k / MRR of hybrid (no reranker) and of each pool size; rerank latency (mean, p95);
peak memory; top reranker score for answerable vs out-of-corpus questions (abstention signal).
Results are merged into eval/retrieval_rerank.json under "<dense>+<reranker>".

Usage:
    uv run python -m aides_rag.retrieval.bench_rerank --dense e5-small --reranker mminilm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aides_rag.index.chunks import read_corpus
from aides_rag.retrieval.bench_dense import peak_memory_gb
from aides_rag.retrieval.bm25 import BM25
from aides_rag.retrieval.dense import MODELS, DenseIndex
from aides_rag.retrieval.eval_recall import SPLITS, score_rankings
from aides_rag.retrieval.hybrid import aids_in_order, hybrid_chunks
from aides_rag.retrieval.reranker import RERANKERS, Reranker

POOLS = (10, 20, 50)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", choices=sorted(MODELS), default="e5-small")
    ap.add_argument("--reranker", choices=sorted(RERANKERS), required=True)
    ap.add_argument("--pools", type=int, nargs="+", default=list(POOLS))
    ap.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    ap.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/retrieval_rerank.json"))
    args = ap.parse_args()

    chunks = read_corpus(args.chunks)
    texts = [c["text"] for c in chunks]
    chunk_aids = np.array([c["id_aid"] for c in chunks])
    golden = read_corpus(args.golden)
    bm25 = BM25(texts)
    dense = DenseIndex(args.dense, texts)
    t0 = time.perf_counter()
    reranker = Reranker(args.reranker)
    load_s = time.perf_counter() - t0

    max_pool = max(args.pools)
    rankings: dict[str, dict[str, list[str]]] = {"hybrid": {}, **{f"rerank@{p}": {} for p in args.pools}}
    latency: dict[int, list[float]] = {p: [] for p in args.pools}
    top_score: dict[str, float] = {}
    for q in golden:
        fused = hybrid_chunks(bm25.scores(q["question"]), dense.scores(q["question"]), depth=max(max_pool, 50))
        rankings["hybrid"][q["question"]] = aids_in_order(fused, chunk_aids)
        for pool in args.pools:
            t0 = time.perf_counter()
            reranked, scores = reranker.rerank(q["question"], fused[:pool], texts)
            latency[pool].append((time.perf_counter() - t0) * 1000)
            # chunks beyond the pool keep their hybrid order, so Recall@50 stays comparable
            rankings[f"rerank@{pool}"][q["question"]] = aids_in_order(reranked + fused[pool:], chunk_aids)
            if pool == max_pool:
                top_score[q["id"]] = float(scores[0]) if len(scores) else float("-inf")

    quality = {}
    for split in SPLITS:
        qs = [q for q in golden if q["split"] == split and q["gold_docs"]]
        quality[split] = {name: score_rankings(qs, lambda question, r=r: r[question]) for name, r in rankings.items()}

    answerable = [top_score[q["id"]] for q in golden if q["gold_docs"]]
    unanswerable = [top_score[q["id"]] for q in golden if q["category"] == "out_of_corpus"]
    result = {
        "dense": MODELS[args.dense].name,
        "reranker": reranker.name,
        "reranker_revision": reranker.revision,
        "reranker_load_seconds": round(load_s, 1),
        "peak_memory_gb": peak_memory_gb(),
        "rerank_ms": {str(p): {"mean": round(float(np.mean(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
                      for p, v in latency.items()},
        "top_reranker_score": {"answerable_mean": round(float(np.mean(answerable)), 3),
                               "answerable_min": round(float(np.min(answerable)), 3),
                               "out_of_corpus_mean": round(float(np.mean(unanswerable)), 3),
                               "out_of_corpus_max": round(float(np.max(unanswerable)), 3)},
        "quality": quality,
    }
    out = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    out[f"{args.dense}+{args.reranker}"] = result
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"{args.dense} + {reranker.name} @ {reranker.revision}  peak RAM={result['peak_memory_gb']} GB")
    print(f"rerank latency: {result['rerank_ms']}")
    print(f"top reranker score: {result['top_reranker_score']}")
    for split in SPLITS:
        print(f"[{split}]")
        for name, res in quality[split].items():
            s = res["summary"]
            misses = [f"{p['id']}:{list(p['gold_ranks'].values())}" for p in res["per_question"]
                      if not all(r and r <= 5 for r in p["gold_ranks"].values())]
            print(f"  {name:<10} R@1={s['recall@1']:<6} R@5={s['recall@5']:<6} MRR={s['mrr']:<6} misses@5={misses}")


if __name__ == "__main__":
    main()
