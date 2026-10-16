"""Systems that answer a question, behind one contract so eval/run_eval.py can score any of them.

Every system is a class with `answer(question: str) -> dict` returning the keys of `OUTPUT_KEYS`:

    answer         str | None   final answer text (None while there is no LLM)
    cited_sources  list[str]    id_aid values cited in the answer
    retrieved_ids  list[str]    id_aid values retrieved, best first
    tokens_in      int | None   prompt tokens (None without an LLM)
    tokens_out     int | None   completion tokens
    tool_calls     list[dict]   one {"name": ..., "args": ...} per tool call (empty without an agent)
    stop_reason    str | None   why generation stopped (e.g. "stop", "length", "tool_use")

Latency is measured by the runner, not by the system. To add a system, write a module in this
package and register it in `SYSTEMS` as "module:Class".
"""

from __future__ import annotations

import importlib

OUTPUT_KEYS = ("answer", "cited_sources", "retrieved_ids", "tokens_in", "tokens_out", "tool_calls", "stop_reason")

SYSTEMS = {
    "bm25": "aides_rag.systems.bm25_only:Bm25Only",
    "hybrid": "aides_rag.systems.hybrid:Hybrid",
}


def load_system(name: str, **kwargs):
    """Instantiate a registered system; heavy dependencies are only imported for the one asked."""
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}, choose from: {', '.join(SYSTEMS)}")
    module, cls = SYSTEMS[name].split(":")
    return getattr(importlib.import_module(module), cls)(**kwargs)
