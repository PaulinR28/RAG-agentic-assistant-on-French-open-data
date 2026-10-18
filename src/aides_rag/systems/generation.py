"""Generating systems (workstream B): an LLM answers from the retrieved aids, with validated citations.

- `Rag`: one forced search with the raw question, then one JSON answer (baseline).
- `Agent`: the model searches with its own queries and filters, reads full aids, then answers.

The model comes from the environment (.env): LLM_BACKEND=ollama|openrouter and LLM_MODEL, so the
same registered system scores the local 4B or an API model, e.g.
    LLM_BACKEND=openrouter LLM_MODEL=anthropic/claude-haiku-5.5 uv run python eval/run_eval.py --system agent

`retrieved_ids`: rag = the k aids of its search, best first; agent = every aid its tools returned,
in order of discovery (can exceed k when it searches twice).
"""

from __future__ import annotations

from aides_rag.agent.agent import answer
from aides_rag.llm.client import LLMClient


class _Generation:
    mode = ""

    def __init__(self, backend: str | None = None, model: str | None = None):
        self.llm = LLMClient.from_env(backend, model)

    def answer(self, question: str) -> dict:
        return answer(question, mode=self.mode, llm=self.llm)


class Rag(_Generation):
    mode = "rag"


class Agent(_Generation):
    mode = "agent"
