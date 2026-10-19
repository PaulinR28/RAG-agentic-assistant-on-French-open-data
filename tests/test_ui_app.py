"""The UI end to end with a fake `answer()`: ask, see the result, reopen it from the history after a restart."""

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

from aides_rag.agent import agent  # noqa: E402
from aides_rag.llm import client  # noqa: E402
from aides_rag.ui import history  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "aides_rag" / "ui" / "app.py")


def fake_row(question, mode="agent", llm=None, full_cards=None, **_):
    return {"id": None, "question": question, "answer": "Le Prêt Vert peut convenir.", "cited_sources": ["411"],
            "abstained": False, "retrieved_ids": ["411", "239"], "tokens_in": 3000, "tokens_out": 120,
            "latency_ms": 1234, "tool_calls": [{"name": "search_aids", "args": {"query": "énergie"},
                                                "latency_ms": 80, "result_ids": ["411", "239"], "forced": False}],
            "n_llm_calls": 2, "stop_reason": "final_answer", "cost_usd": 0.0, "model": llm.model,
            "config": mode + ("+full_cards" if full_cards else ""), "valid_output": True, "error": None,
            "llm_calls": [{"phase": "answer", "tokens_in": 3000, "tokens_out": 120, "latency_ms": 1100,
                           "cost_usd": 0.0, "finish_reason": "stop"}]}


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_PATH", tmp_path / "ui_history.jsonl")
    monkeypatch.setattr(agent, "answer", fake_row)
    monkeypatch.setattr(client.LLMClient, "__post_init__", lambda self: None)  # no Ollama needed
    return tmp_path / "ui_history.jsonl"


def test_ask_shows_answer_and_saves_settings(ui):
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=60).run()
    at.toggle[0].set_value(True)
    at.text_area[0].input("Aide pour réduire l'énergie à Lyon ?")
    at.button[0].click().run()
    assert not at.exception
    assert any("Prêt Vert" in m.value for m in at.markdown)
    [entry] = history.load(ui)
    assert entry["settings"]["full_cards"] is True and entry["row"]["config"].endswith("+full_cards")


def test_history_reopens_after_restart_without_rerunning(ui, monkeypatch):
    history.save(fake_row("Question d'hier", llm=client.LLMClient("ollama", "qwen")),
                 {"mode": "agent", "backend": "ollama", "model": "qwen", "full_cards": False}, ui)
    monkeypatch.setattr(agent, "answer", lambda *a, **k: pytest.fail("reopening must not re-run the model"))
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=60).run()  # fresh session = app restarted
    [past] = [b for b in at.sidebar.button if b.label == "Question d'hier"]
    past.click().run()
    assert not at.exception
    assert any("Prêt Vert" in m.value for m in at.markdown)
