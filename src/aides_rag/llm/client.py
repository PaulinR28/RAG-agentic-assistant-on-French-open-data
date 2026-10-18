"""One interface over the two generation backends: local Ollama and the OpenRouter API.

Both return an `LLMResponse` with the same fields (text, tool calls, token counts, latency, cost),
so the agent and the tracking never depend on the backend. Messages use the OpenAI chat format
(roles system/user/assistant/tool); each backend converts what it needs.

- ollama: native API through the `ollama` SDK. Its OpenAI-compatible layer ignores `think=False`
  and `num_ctx` (smoke test, 2026-10-16), and both matter on the 8 GB demo machine.
- openrouter: OpenAI-compatible API through the `openai` SDK; `usage.cost` is the billed cost.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

NUM_CTX = 8192  # context window of the local model: prompt + 5 chunks + tool results fit in ~4k
KEEP_ALIVE = "30m"  # keep the model in memory between questions (first load takes ~6 s)
TIMEOUT_S = 180
DEFAULT_MODELS = {"ollama": "qwen3:4b-instruct-2507-q4_K_M", "openrouter": "anthropic/claude-haiku-5.5"}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float
    finish_reason: str  # "stop", "length" or "tool_calls"

    def as_message(self) -> dict:
        """The assistant turn to append to the conversation before sending tool results back."""
        msg = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                                 for tc in self.tool_calls]
        return msg


@dataclass
class LLMClient:
    backend: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 700
    _sdk: object = field(default=None, repr=False)

    @classmethod
    def from_env(cls, backend: str | None = None, model: str | None = None) -> LLMClient:
        backend = backend or os.environ.get("LLM_BACKEND", "ollama")
        if backend not in DEFAULT_MODELS:
            raise ValueError(f"LLM_BACKEND must be one of {list(DEFAULT_MODELS)}, got {backend!r}")
        return cls(backend, model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS[backend])

    def __post_init__(self) -> None:
        if self.backend == "ollama":
            import ollama
            self._sdk = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), timeout=TIMEOUT_S)
        else:
            from openai import OpenAI
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is missing: add it to .env (see .env.example)")
            self._sdk = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=TIMEOUT_S)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, schema: dict | None = None,
             max_tokens: int | None = None) -> LLMResponse:
        """One model call. `tools` = OpenAI-style tool specs; `schema` = JSON schema the reply must follow;
        `max_tokens` overrides the client's default output cap for this call."""
        start = time.perf_counter()
        call = self._ollama if self.backend == "ollama" else self._openrouter
        resp = call(messages, tools, schema, max_tokens or self.max_tokens)
        resp.latency_s = round(time.perf_counter() - start, 3)
        return resp

    def _ollama(self, messages, tools, schema, max_tokens) -> LLMResponse:
        r = self._sdk.chat(model=self.model, messages=[_to_ollama(m) for m in messages], tools=tools, format=schema,
                           think=False, keep_alive=KEEP_ALIVE,
                           options={"num_ctx": NUM_CTX, "temperature": self.temperature, "num_predict": max_tokens})
        calls = [ToolCall(f"call_{i}", tc.function.name, dict(tc.function.arguments or {}))
                 for i, tc in enumerate(r.message.tool_calls or [])]
        return LLMResponse(r.message.content or "", calls, r.prompt_eval_count or 0, r.eval_count or 0, 0.0, 0.0,
                           "tool_calls" if calls else (r.done_reason or "stop"))

    def _openrouter(self, messages, tools, schema, max_tokens) -> LLMResponse:
        # No reasoning phase for any API model, like the local instruct model: comparisons measure model size and
        # provider, not thinking. The providers serving Qwen3 hybrids ignore effort "none" (measured: 307 reasoning
        # tokens), but obey Qwen's own "/no_think" switch in the system prompt (1 token).
        messages = [_to_openai(m) for m in messages]
        if "qwen3" in self.model and messages and messages[0]["role"] == "system":
            messages[0] = {**messages[0], "content": messages[0]["content"] + "\n/no_think"}
        kwargs = {"tools": tools} if tools else {}
        if schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "Answer", "strict": True, "schema": schema}}
        r = self._sdk.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature,
                                              max_tokens=max_tokens, extra_body={"reasoning": {"effort": "none"}}, **kwargs)
        choice, usage = r.choices[0], r.usage
        calls = [ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}"))
                 for tc in (choice.message.tool_calls or [])]
        return LLMResponse(choice.message.content or "", calls, usage.prompt_tokens, usage.completion_tokens, 0.0,
                           float(getattr(usage, "cost", 0.0) or 0.0), "tool_calls" if calls else choice.finish_reason)


def _to_ollama(m: dict) -> dict:
    """Ollama wants tool arguments as dicts and tool results tagged with the tool's name."""
    if m["role"] == "tool":
        return {"role": "tool", "tool_name": m["name"], "content": m["content"]}
    if m.get("tool_calls"):
        return {"role": "assistant", "content": m["content"],
                "tool_calls": [{"function": {"name": tc["function"]["name"],
                                             "arguments": json.loads(tc["function"]["arguments"])}}
                               for tc in m["tool_calls"]]}
    return m


def _to_openai(m: dict) -> dict:
    """OpenAI links a tool result to its call by id; the tool name is not part of the message."""
    if m["role"] == "tool":
        return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
    return m
