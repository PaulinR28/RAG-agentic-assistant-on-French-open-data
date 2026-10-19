"""Systems that answer a question, behind one contract so eval/run_eval.py can score any of them.

Every system is a class with `answer(question: str) -> dict` returning the keys of `OUTPUT_KEYS`:

    answer         str | None   final answer text (None while there is no LLM)
    cited_sources  list[str]    id_aid values cited in the answer
    retrieved_ids  list[str]    id_aid values retrieved, best first
    tokens_in      int | None   prompt tokens (None without an LLM)
    tokens_out     int | None   completion tokens
    tool_calls     list[dict]   one {"name": ..., "args": ...} per tool call (empty without an agent)
    stop_reason    str | None   why generation stopped (e.g. "stop", "length", "tool_use")
    abstained      bool | None  the system said the corpus does not hold the answer (None without an LLM)
    cost_usd       float | None billed cost of the answer (0 for a local model, None without an LLM)
    model          str | None   generating model, e.g. "qwen3:4b-instruct-2507-q4_K_M"

Latency is measured by the runner, not by the system. To add a system, write a module in this
package and register it in `SYSTEMS` as "module:Class".
"""

from __future__ import annotations

import importlib

OUTPUT_KEYS = ("answer", "cited_sources", "retrieved_ids", "tokens_in", "tokens_out", "tool_calls", "stop_reason",
               "abstained", "cost_usd", "model")

SYSTEMS = {
    "bm25": "aides_rag.systems.bm25_only:Bm25Only",
    "hybrid": "aides_rag.systems.hybrid:Hybrid",
    "hybrid_rerank": "aides_rag.systems.hybrid:HybridRerank",
    "rag": "aides_rag.systems.generation:Rag",
    "agent": "aides_rag.systems.generation:Agent",
}


def load_system(name: str, **kwargs):
    """Instantiate a registered system; heavy dependencies are only imported for the one asked."""
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}, choose from: {', '.join(SYSTEMS)}")
    module, cls = SYSTEMS[name].split(":")
    return getattr(importlib.import_module(module), cls)(**kwargs)
