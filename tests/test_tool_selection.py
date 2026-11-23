from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.agent import AgentResponse
from evaluation.tool_selection import (
    EvalCase,
    evaluate_tool_selection,
    format_summary,
    load_cases,
    summarise,
)

CASES = Path("datasets/instructions/tool_selection_eval.jsonl")
SFT = Path("datasets/instructions/tiny50_sft.jsonl")


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


class FakeAgent:
    def __init__(self, responses):
        self.responses = responses

    def chat(self, prompt):
        return self.responses[prompt]


def _response(names, ok=True, errors=None):
    return AgentResponse(
        text="done",
        iterations=1,
        tool_calls=[{"name": name, "arguments": {}, "ok": ok, "content": "", "error": None} for name in names],
        errors=errors or [],
    )


def test_cases_load_and_are_unique():
    cases = load_cases(CASES)
    assert len(cases) == 25
    assert len({case.id for case in cases}) == 25
    assert {case.category for case in cases} >= {
        "calculator",
        "local_search",
        "read_document",
        "list_files",
        "retrieve_memory",
        "save_memory",
        "no_tool",
        "unavailable_tool",
        "multi_tool",
    }


def test_eval_prompts_are_held_out_from_sft():
    sft_prompts = set()
    for line in SFT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        for message in sample["messages"]:
            if message["role"] == "user":
                sft_prompts.add(_normalise(message["content"]))
    for case in load_cases(CASES):
        normalised = _normalise(case.prompt)
        assert normalised not in sft_prompts, f"evaluation prompt leaked into SFT: {case.prompt!r}"
        assert not any(normalised in prompt for prompt in sft_prompts), (
            f"evaluation prompt is a substring of an SFT prompt: {case.prompt!r}"
        )


def test_metrics_with_scripted_responses():
    cases = [
        EvalCase("a", "Compute 2+2", ["calculator"], "calculator"),
        EvalCase("b", "Compute 3+3", ["calculator"], "calculator"),
        EvalCase("c", "Say hi", [], "no_tool"),
    ]
    agent = FakeAgent(
        {
            "Compute 2+2": _response(["calculator"]),
            "Compute 3+3": _response([]),
            "Say hi": _response(["calculator"]),
        }
    )
    report = evaluate_tool_selection(agent, cases)
    summary = report["summary"]
    assert summary["cases"] == 3
    assert summary["first_tool_accuracy"] == pytest.approx(1 / 3)
    assert summary["exact_set_accuracy"] == pytest.approx(1 / 3)
    assert summary["missing_tool_rate"] == pytest.approx(1 / 2)
    assert summary["no_tool_false_positive_rate"] == pytest.approx(1.0)
    assert summary["invalid_call_rate"] == 0.0
    assert report["results"][0]["exact_set_ok"] is True


def test_invalid_and_failed_calls_counted():
    cases = [EvalCase("a", "Search x", ["local_search"], "local_search")]
    failed = AgentResponse(
        text="",
        iterations=1,
        tool_calls=[{"name": "local_search", "arguments": {}, "ok": False, "content": "", "error": "bad args"}],
        errors=["tool call is not valid JSON"],
    )
    report = evaluate_tool_selection(FakeAgent({"Search x": failed}), cases)
    assert report["summary"]["invalid_call_rate"] == 1.0
    assert report["summary"]["cases_with_parse_errors"] == 1
    assert report["results"][0]["called"] == ["local_search"]


def test_exact_set_is_order_insensitive():
    cases = [EvalCase("m", "both", ["local_search", "calculator"], "multi_tool")]
    response = _response(["calculator", "local_search"])
    result = evaluate_tool_selection(FakeAgent({"both": response}), cases)
    assert result["summary"]["exact_set_accuracy"] == 1.0
    assert result["summary"]["first_tool_accuracy"] == 0.0  # calculator != local_search


def test_bad_case_files_rejected(tmp_path):
    duplicate = tmp_path / "dup.jsonl"
    duplicate.write_text(
        "\n".join(
            [
                json.dumps({"id": "x", "prompt": "one", "expected_tools": []}),
                json.dumps({"id": "x", "prompt": "two", "expected_tools": []}),
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_cases(duplicate)

    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(
        json.dumps({"id": "y", "prompt": "p", "expected_tools": ["launch_rocket"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown expected tool"):
        load_cases(unknown)


def test_summary_and_format_are_consistent():
    results = [
        {
            "first_tool_ok": True,
            "exact_set_ok": True,
            "invalid_call": False,
            "missed_tool": False,
            "false_positive": False,
            "expected": ["calculator"],
            "category": "calculator",
        }
    ]
    from evaluation.tool_selection import CaseResult

    case_result = CaseResult(
        id="a",
        category="calculator",
        prompt="p",
        expected=["calculator"],
        called=["calculator"],
        calls=[{"name": "calculator", "arguments": {}, "ok": True}],
        first_tool_ok=True,
        exact_set_ok=True,
        invalid_call=False,
        missed_tool=False,
        false_positive=False,
        text="",
        errors=[],
        iterations=1,
        tokens_generated=1,
    )
    summary = summarise([case_result])
    assert summary["first_tool_accuracy"] == 1.0
    text = format_summary({"summary": summary})
    assert "first-tool accuracy" in text
