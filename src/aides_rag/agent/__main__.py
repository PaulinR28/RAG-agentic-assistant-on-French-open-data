"""Ask one question from the terminal and see the answer with its trace.

    uv run python -m aides_rag.agent "Quel est le taux du CIR ?"
    uv run python -m aides_rag.agent --qid q007 --mode rag     # a dev golden question, with its expected answer
    uv run python -m aides_rag.agent "..." --backend openrouter --json

--qid only accepts dev questions: held-out questions stay unseen until the final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aides_rag.agent.agent import MODES, answer
from aides_rag.index.chunks import read_corpus
from aides_rag.llm.client import LLMClient

GOLDEN = Path(__file__).resolve().parents[3] / "eval" / "golden.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--qid", help="dev golden question id, e.g. q007")
    ap.add_argument("--mode", choices=MODES, default="agent")
    ap.add_argument("--backend", choices=["ollama", "openrouter"])
    ap.add_argument("--model")
    ap.add_argument("--json", action="store_true", help="print the full results.jsonl row")
    args = ap.parse_args()

    expected = None
    if args.qid:
        gold = {q["id"]: q for q in read_corpus(GOLDEN)}
        q = gold.get(args.qid) or ap.error(f"unknown question id {args.qid}")
        if q.get("split") != "dev":
            ap.error(f"{args.qid} is a held-out question: it is only run by the final evaluation")
        args.question, expected = q["question"], q
    if not args.question:
        ap.error("give a question or --qid")

    row = answer(args.question, mode=args.mode, qid=args.qid, llm=LLMClient.from_env(args.backend, args.model))
    if args.json:
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return
    print(f"Q: {row['question']}\n\nA: {row['answer'] if row['valid_output'] else '(no valid answer)'}")
    print(f"   citations={row['cited_sources']} abstained={row['abstained']} stop={row['stop_reason']}")
    if row["error"]:
        print(f"   error: {row['error']}")
    print(f"\n{row['model']} | mode={row['config']} | {row['latency_ms'] / 1000:.1f} s | "
          f"tokens in/out {row['tokens_in']}/{row['tokens_out']} | LLM calls {row['n_llm_calls']} | "
          f"cost {row['cost_usd']} USD")
    for tc in row["tool_calls"]:
        print(f"   tool {tc['name']}({tc['args']}) -> {tc['result_ids']} [{tc['latency_ms']} ms]")
    if expected:
        print(f"\nexpected ({expected['expected_behavior']}): {expected['answer'] or '-'} | gold_docs={expected['gold_docs']}")


if __name__ == "__main__":
    main()
