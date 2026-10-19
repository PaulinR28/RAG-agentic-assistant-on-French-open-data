"""Answer one question and return its results.jsonl row.

Modes:
- "rag": one forced search with the raw question, then one structured answer call (baseline).
- "agent": the model chooses its own searches (query, filters) and reads full aids with tools,
  at most MAX_TOOL_CALLS calls, then answers.

Both end with the same answer phase: a JSON reply checked by `parse_answer` (shape + grounding),
with one retry that shows the model its error. Every step is appended to the same conversation,
so Ollama reuses the cached prompt prefix instead of re-reading the aids (~15 s on the M1).
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

from aides_rag.agent.prompts import agent_messages, answer_request, rag_messages
from aides_rag.agent.schema import ANSWER_SCHEMA, Answer, InvalidAnswer, parse_answer
from aides_rag.agent.tools import TOOL_SPECS, run_tool
from aides_rag.agent.trace import Trace
from aides_rag.llm.client import NUM_CTX, LLMClient, ToolCall
from aides_rag.retrieval.search import full_cards_enabled, search, use_full_cards

MODES = ("rag", "agent")
TOP_K = 5
MAX_TOOL_CALLS = 3
TOOL_ROUND_TOKENS = 200  # enough for two tool calls; caps the draft a small model writes instead of "PRÊT"
CONTEXT_BUDGET = NUM_CTX - 1500  # keep room for the answer phase; past NUM_CTX Ollama drops the system prompt
# Qwen3 splits our French text into ~3.5 chars/token (mean 3.49, 5th percentile 3.18, min 2.77 over the 865
# chunks), denser than the 4.0 of chunks.approx_tokens (measured on bge-m3's tokenizer). The guard must not
# undercount, so it uses 3.0: an overestimate for almost every text.
GUARD_CHARS_PER_TOKEN = 3.0
LIMIT_NOTE = "Limite de 3 appels d'outils atteinte : appel non exécuté. Réponds avec les fiches déjà fournies."
FULL_NOTE = "Résultat non affiché : le contexte est plein. Réponds avec les fiches déjà fournies."


def answer(question: str, mode: str = "agent", qid: str | None = None, llm: LLMClient | None = None,
           k: int = TOP_K, full_cards: bool | None = None) -> dict:
    """`full_cards`: show each found aid's whole card instead of its best chunk (UI toggle; None = default, off)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    llm = llm or LLMClient.from_env()
    with use_full_cards(full_cards):
        config = mode + ("+full_cards" if full_cards_enabled() else "")  # recorded in results.jsonl
        return _answer(question, mode, qid, llm, k, Trace(question, llm.model, config, qid))


def _answer(question: str, mode: str, qid: str | None, llm: LLMClient, k: int, trace: Trace) -> dict:
    try:
        if mode == "rag":
            messages, hit_limit = _rag(question, k, trace), False
        else:
            messages, hit_limit = _agent(question, llm, trace)
        # Ollama writes the tool specs at the top of the prompt: keeping them in the answer call keeps the
        # prompt prefix identical, so the cached aids are not re-read (measured: 1.2 s vs 21.7 s on the M1)
        keep_tools = TOOL_SPECS if mode == "agent" and llm.backend == "ollama" else None
        ans = _final_answer(llm, messages, trace, keep_tools)
        if ans and hit_limit:
            trace.stop_reason = "max_tool_calls"  # answered, but the agent did not stop searching on its own
        return trace.row(ans)
    except Exception as e:  # one bad question must not stop an evaluation run
        trace.stop_reason = "error"
        return trace.row(None, error=f"{type(e).__name__}: {e}")


def _rag(question: str, k: int, trace: Trace) -> list[dict]:
    start = time.perf_counter()
    hits = search(question, k=k)
    trace.add_tool("search_aids", {"query": question, "k": k}, time.perf_counter() - start, [h.id_aid for h in hits],
                   forced=True)
    return rag_messages(question, hits)


def _agent(question: str, llm: LLMClient, trace: Trace) -> tuple[list[dict], bool]:
    """Tool loop. Returns the conversation, ready for the answer phase, and whether a call was refused
    because the limit was reached (the model wanted to keep searching)."""
    messages, shown, n_calls, round_, refused = agent_messages(question), set(), 0, 0, False
    while n_calls < MAX_TOOL_CALLS:
        round_ += 1
        resp = llm.chat(messages, tools=TOOL_SPECS, max_tokens=TOOL_ROUND_TOKENS)
        trace.add_llm(resp, f"tools_{round_}")
        calls, forced = resp.tool_calls, False
        if not calls and n_calls == 0:
            # the model wanted to answer without looking (dev q004, q005: it wrongly judged the question out
            # of scope). Search with the raw question for it, as rag mode does, so it never knows less than rag
            calls, forced = [ToolCall("call_forced", "search_aids", {"query": question})], True
        if not calls:
            # the model has what it needs (or refuses). Its free-text draft is dropped: copied into the
            # JSON answer, a wrong draft became a wrong answer (dev q007); the cache still covers the aids
            return messages + [answer_request(question)], False
        messages.append(replace(resp, content="" if forced else resp.content, tool_calls=calls).as_message())
        context = resp.tokens_in + resp.tokens_out
        for tc in calls:
            # every call gets a tool message (the OpenAI API requires one per call id), even when not executed
            if n_calls >= MAX_TOOL_CALLS:
                text, refused = LIMIT_NOTE, True
            else:
                start, seen_before = time.perf_counter(), set(shown)
                text, ids = run_tool(tc.name, tc.arguments, shown)
                if context + _guard_tokens(text) > CONTEXT_BUDGET:
                    text, ids = FULL_NOTE, []
                    shown.intersection_update(seen_before)  # not shown after all: give them in full if found again
                trace.add_tool(tc.name, tc.arguments, time.perf_counter() - start, ids, forced=forced)
                n_calls += 1
            context += _guard_tokens(text)
            messages.append({"role": "tool", "name": tc.name, "tool_call_id": tc.id, "content": text})
    return messages + [answer_request(question)], refused


def _guard_tokens(text: str) -> int:
    """Upper-side estimate of the LLM tokens a tool result adds (see GUARD_CHARS_PER_TOKEN)."""
    return math.ceil(len(text) / GUARD_CHARS_PER_TOKEN)


def _final_answer(llm: LLMClient, messages: list[dict], trace: Trace, tools: list[dict] | None = None) -> Answer | None:
    for attempt in ("answer", "retry"):
        resp = llm.chat(messages, tools=tools, schema=ANSWER_SCHEMA)
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
