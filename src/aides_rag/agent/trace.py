"""What the system did for one question: LLM calls, tool calls, tokens, latency, cost, stop reason.

`Trace.row()` is one line of eval/results.jsonl, the format agreed with workstream C (evaluation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from aides_rag.agent.schema import Answer
from aides_rag.llm.client import LLMResponse

STOP_REASONS = ("final_answer", "abstained", "max_tool_calls", "invalid_output", "error")


@dataclass
class Trace:
    question: str
    model: str
    config: str  # e.g. "agent" or "rag", plus the retriever once workstream A ships variants
    qid: str | None = None
    llm_calls: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)  # aid level, deduplicated, in order of discovery
    stop_reason: str | None = None
    _start: float = field(default_factory=time.perf_counter, repr=False)

    def add_llm(self, resp: LLMResponse, phase: str) -> None:
        self.llm_calls.append({"phase": phase, "tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out,
                               "latency_ms": round(resp.latency_s * 1000), "cost_usd": resp.cost_usd,
                               "finish_reason": resp.finish_reason})

    def add_tool(self, name: str, args: dict, latency_s: float, result_ids: list[str], forced: bool = False) -> None:
        """`forced`: run by the code (rag mode, or an agent that answered without searching), not chosen by the model."""
        self.tool_calls.append({"name": name, "args": args, "latency_ms": round(latency_s * 1000),
                                "result_ids": result_ids, "forced": forced})
        self.retrieved_ids.extend(i for i in result_ids if i not in self.retrieved_ids)

    def row(self, answer: Answer | None, error: str | None = None) -> dict:
        assert self.stop_reason in STOP_REASONS, self.stop_reason
        return {
            "id": self.qid,
            "question": self.question,
            "answer": answer.answer if answer else "",
            "cited_sources": answer.citations if answer else [],
            "abstained": answer.abstained if answer else None,
            "retrieved_ids": self.retrieved_ids,
            "tokens_in": sum(c["tokens_in"] for c in self.llm_calls),
            "tokens_out": sum(c["tokens_out"] for c in self.llm_calls),
            "latency_ms": round((time.perf_counter() - self._start) * 1000),
            "tool_calls": self.tool_calls,
            "n_llm_calls": len(self.llm_calls),
            "stop_reason": self.stop_reason,
            "cost_usd": round(sum(c["cost_usd"] for c in self.llm_calls), 6),
            "model": self.model,
            "config": self.config,
            "valid_output": answer is not None,
            "error": error,
            "llm_calls": self.llm_calls,
        }
