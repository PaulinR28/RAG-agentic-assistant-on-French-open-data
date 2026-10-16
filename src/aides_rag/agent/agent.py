"""Answer one question and return its results.jsonl row.

Modes:
- "rag": one forced search with the raw question, then one structured answer call (baseline).
- "agent": the model chooses its own searches and reads full aids with tools, then answers (step 4).

Both end with the same answer phase: a JSON reply checked by `parse_answer` (shape + grounding),
with one retry that shows the model its error. The retry is appended to the same conversation,
so Ollama reuses the cached prompt prefix instead of re-reading the aids (~15 s on the M1).
"""

from __future__ import annotations

import time

from aides_rag.agent.prompts import rag_messages
from aides_rag.agent.schema import ANSWER_SCHEMA, Answer, InvalidAnswer, parse_answer
from aides_rag.agent.trace import Trace
from aides_rag.llm.client import LLMClient
from aides_rag.retrieval.search import search

MODES = ("rag", "agent")
TOP_K = 5


def answer(question: str, mode: str = "agent", qid: str | None = None, llm: LLMClient | None = None,
           k: int = TOP_K) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    llm = llm or LLMClient.from_env()
    trace = Trace(question, llm.model, mode, qid)
    try:
        if mode == "rag":
            messages = _rag(question, k, trace)
        else:
            raise NotImplementedError("agent mode comes in step 4")
        ans = _final_answer(llm, messages, trace)
        return trace.row(ans)
    except Exception as e:  # one bad question must not stop an evaluation run
        trace.stop_reason = "error"
        return trace.row(None, error=f"{type(e).__name__}: {e}")


def _rag(question: str, k: int, trace: Trace) -> list[dict]:
    start = time.perf_counter()
    hits = search(question, k=k)
    trace.add_tool("search_aids", {"query": question, "k": k}, time.perf_counter() - start, [h.id_aid for h in hits])
    return rag_messages(question, hits)


def _final_answer(llm: LLMClient, messages: list[dict], trace: Trace) -> Answer | None:
    for attempt in ("answer", "retry"):
        resp = llm.chat(messages, schema=ANSWER_SCHEMA)
        trace.add_llm(resp, attempt)
        try:
            ans = parse_answer(resp.content, trace.retrieved_ids)
        except InvalidAnswer as e:
            messages = messages + [{"role": "assistant", "content": resp.content},
                                   {"role": "user", "content": f"Réponse invalide : {e} Corrige ta réponse JSON."}]
            continue
        trace.stop_reason = "abstained" if ans.abstained else "final_answer"
        return ans
    trace.stop_reason = "invalid_output"
    return None
