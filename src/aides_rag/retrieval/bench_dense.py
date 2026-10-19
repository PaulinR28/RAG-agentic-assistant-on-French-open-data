"""Benchmark embedding models on this laptop: retrieval quality, speed and memory, dense alone and hybrid.

For each model (run one model per process, so memory figures are clean):
  - build time: model load + embedding of all chunks (or loading cached vectors)
  - memory: process peak working set
  - query latency: embedding one question + scoring all chunks (mean and p95 over the golden questions)
  - Recall@k / MRR per split (dev, heldout) for dense alone and for hybrid BM25 + dense (RRF)
  - top cosine score for answerable vs out-of-corpus questions (input for an abstention threshold later)
Results are merged into eval/retrieval_dense.json, one entry per model.

Usage:
    uv run python -m aides_rag.retrieval.bench_dense --model e5-small
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil

from aides_rag.index.chunks import read_corpus
from aides_rag.retrieval.bm25 import BM25
from aides_rag.retrieval.dense import MODELS, DenseIndex
from aides_rag.retrieval.eval_recall import SPLITS, score_rankings
from aides_rag.retrieval.hybrid import rank_aids, rrf


def peak_memory_gb() -> float:
    info = psutil.Process().memory_info()
    return round(getattr(info, "peak_wset", info.rss) / 1e9, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    ap.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/retrieval_dense.json"))
    args = ap.parse_args()

    chunks = read_corpus(args.chunks)
    texts = [c["text"] for c in chunks]
    chunk_aids = np.array([c["id_aid"] for c in chunks])
    golden = read_corpus(args.golden)
    bm25 = BM25(texts)

    t0 = time.perf_counter()
    dense = DenseIndex(args.model, texts)
    build_s = time.perf_counter() - t0

    latencies, dense_scores = [], {}
    for q in golden:
        t0 = time.perf_counter()
        dense_scores[q["id"]] = dense.scores(q["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
    by_question = {q["question"]: q["id"] for q in golden}

    def dense_rank(question: str) -> list[str]:
        return rank_aids(dense_scores[by_question[question]], chunk_aids)

    def hybrid_rank(question: str) -> list[str]:
        return rrf([rank_aids(bm25.scores(question), chunk_aids, positive_only=True), dense_rank(question)])

    def bm25_rank(question: str) -> list[str]:
        return rank_aids(bm25.scores(question), chunk_aids, positive_only=True)

    quality = {}
    for split in SPLITS:
        qs = [q for q in golden if q["split"] == split and q["gold_docs"]]
        quality[split] = {name: score_rankings(qs, fn) for name, fn in
                          [("bm25", bm25_rank), ("dense", dense_rank), ("hybrid", hybrid_rank)]}

    top_cos = {q["id"]: float(dense_scores[q["id"]].max()) for q in golden}
    answerable = [top_cos[q["id"]] for q in golden if q["gold_docs"]]
    unanswerable = [top_cos[q["id"]] for q in golden if q["category"] == "out_of_corpus"]

    result = {
        "model": dense.spec.name,
        "revision": dense.revision,
        "dim": int(dense.vectors.shape[1]),
        "max_seq_length": dense.spec.max_seq_length,
        "vectors_from_cache": dense.from_cache,
        "build_seconds": round(build_s, 1),
        "peak_memory_gb": peak_memory_gb(),
        "query_ms_mean": round(float(np.mean(latencies)), 1),
        "query_ms_p95": round(float(np.percentile(latencies, 95)), 1),
        "top_cosine": {"answerable_mean": round(float(np.mean(answerable)), 3),
                       "answerable_min": round(float(np.min(answerable)), 3),
                       "out_of_corpus_mean": round(float(np.mean(unanswerable)), 3),
                       "out_of_corpus_max": round(float(np.max(unanswerable)), 3)},
        "quality": quality,
    }

    out = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    out[args.model] = result
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"{args.model}: {dense.spec.name} @ {dense.revision}  dim={result['dim']}  "
          f"build={result['build_seconds']}s (cache={dense.from_cache})  peak RAM={result['peak_memory_gb']} GB  "
          f"query={result['query_ms_mean']} ms (p95 {result['query_ms_p95']})")
    print(f"top cosine: {result['top_cosine']}")
    for split in SPLITS:
        print(f"[{split}]")
        for name, res in quality[split].items():
            s = res["summary"]
            misses = [f"{p['id']}:{list(p['gold_ranks'].values())}" for p in res["per_question"]
                      if not all(r and r <= 5 for r in p["gold_ranks"].values())]
            print(f"  {name:<7} R@1={s['recall@1']:<6} R@5={s['recall@5']:<6} R@10={s['recall@10']:<6} "
                  f"MRR={s['mrr']:<6} misses@5={misses}")


if __name__ == "__main__":
    main()
