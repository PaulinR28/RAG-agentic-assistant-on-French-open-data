"""The judge's plumbing, with a fake model: grading, error rows, per-split scores, the --judge option."""

import json
from pathlib import Path

from aides_rag.evaluation import judge
from aides_rag.evaluation.run_eval import main
from aides_rag.llm.client import LLMResponse

GOLDEN = Path(__file__).resolve().parents[1] / "eval/golden.jsonl"
GOLD = {"id": "q001", "answer": "30 %", "rationale": "montant", "expected_behavior": "answer"}


class FakeJudge:
    model = "fake-judge"

    def __init__(self, verdict="correct"):
        self.verdict, self.prompts = verdict, []

    def chat(self, messages, schema=None, max_tokens=None, tools=None):
        assert schema == judge.Verdict.model_json_schema()
        self.prompts.append(messages[-1]["content"])
        return LLMResponse(json.dumps({"correctness": self.verdict, "behavior_ok": True, "reason": "ok"}),
                           [], 10, 5, 0.1, 0.002, "stop")


def test_grade_sees_reference_and_answer():
    llm = FakeJudge("partial")
    row = {"id": "q001", "split": "dev", "category": "factual", "question": "Taux du CIR ?", "answer": "30 % environ",
           "abstained": False}
    j = judge.judge_row(llm, row, GOLD)
    assert j["correctness"] == "partial" and j["behavior_ok"] and j["judge_cost_usd"] == 0.002
    assert "30 %" in llm.prompts[0] and "30 % environ" in llm.prompts[0] and "Expected behaviour: answer" in llm.prompts[0]


def test_row_without_answer_is_wrong_without_a_call():
    llm = FakeJudge()
    j = judge.judge_row(llm, {"id": "q001", "split": "dev", "question": "?", "answer": ""}, GOLD)
    assert j["correctness"] == "wrong" and not j["behavior_ok"] and llm.prompts == []


def test_scores_are_split_by_dev_and_heldout():
    judged = [{"split": "dev", "correctness": "correct", "behavior_ok": True, "judge_cost_usd": 0.001, "judge_model": "m"},
              {"split": "heldout", "correctness": "wrong", "behavior_ok": False, "judge_cost_usd": 0.001, "judge_model": "m"},
              {"split": "heldout", "correctness": "partial", "behavior_ok": True, "judge_cost_usd": 0.001, "judge_model": "m"}]
    s = judge.summarize(judged)
    assert s["judged_correctness"]["heldout"] == {"n": 2, "correct": 0, "partial": 1, "wrong": 1, "accuracy": 0.0}
    assert s["judged_correctness"]["dev"]["accuracy"] == 1.0
    assert s["expected_behavior_respected"]["heldout"]["respected"] == 1
    assert s["judged_correctness"]["judge_cost_usd"] == 0.003


def test_run_eval_judge_option(tmp_path, monkeypatch):
    def answered(self, question):
        return {"answer": "Oui", "cited_sources": [], "retrieved_ids": [], "tokens_in": 1, "tokens_out": 1,
                "tool_calls": [], "stop_reason": "final_answer"}

    monkeypatch.setattr("aides_rag.systems.bm25_only.Bm25Only.answer", answered)
    monkeypatch.setattr(judge, "judge_client", lambda: FakeJudge())
    summary = main(["--questions", str(GOLDEN), "--out", str(tmp_path), "--limit", "3", "--system", "bm25", "--judge"])
    assert summary["judged_correctness"]["all"]["correct"] == 3
    assert len((tmp_path / "judged.jsonl").read_text(encoding="utf-8").splitlines()) == 3
