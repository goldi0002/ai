"""Held-out tool-selection evaluation.

Each labelled case names the tools a correct answer requires. The evaluator
runs the real agent loop on the prompt, records every tool call the model
actually made (including failed and malformed ones), and reports:

* **first-tool accuracy** — the first executed tool matches the first expected
  tool (or no tool was called when none is expected);
* **exact-set accuracy** — the set of called tools equals the expected set;
* **invalid-call rate** — cases with a failed call, unknown tool or parse error;
* **no-tool false positives** — direct-answer prompts that called a tool;
* **missed-tool rate** — prompts that needed a tool but produced none.

Prompts in ``tool_selection_eval.jsonl`` are deliberately disjoint from the SFT
file; ``tests/test_tool_selection.py`` enforces that.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

DEFAULT_CASES = "datasets/instructions/tool_selection_eval.jsonl"
VALID_TOOLS = {
    "calculator",
    "local_search",
    "read_document",
    "list_files",
    "retrieve_memory",
    "save_memory",
}


@dataclass
class EvalCase:
    id: str
    prompt: str
    expected_tools: List[str]
    category: str = "uncategorised"
    notes: str = ""


@dataclass
class CaseResult:
    id: str
    category: str
    prompt: str
    expected: List[str]
    called: List[str]
    calls: List[Dict[str, Any]]
    first_tool_ok: bool
    exact_set_ok: bool
    invalid_call: bool
    missed_tool: bool
    false_positive: bool
    text: str
    errors: List[str]
    iterations: int
    tokens_generated: int
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_cases(path: str | Path = DEFAULT_CASES) -> List[EvalCase]:
    cases: List[EvalCase] = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for key in ("id", "prompt", "expected_tools"):
            if key not in record:
                raise ValueError(f"{path}:{line_number}: missing {key!r}")
        if record["id"] in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {record['id']!r}")
        seen.add(record["id"])
        expected = list(record["expected_tools"])
        if any(tool not in VALID_TOOLS for tool in expected):
            raise ValueError(f"{path}:{line_number}: unknown expected tool in {expected}")
        if not record["prompt"].strip():
            raise ValueError(f"{path}:{line_number}: empty prompt")
        cases.append(
            EvalCase(
                id=record["id"],
                prompt=record["prompt"],
                expected_tools=expected,
                category=record.get("category", "uncategorised"),
                notes=record.get("notes", ""),
            )
        )
    if not cases:
        raise ValueError(f"{path}: no evaluation cases")
    return cases


def _score_case(case: EvalCase, response: Any) -> CaseResult:
    tool_calls = list(getattr(response, "tool_calls", []) or [])
    called = [call["name"] for call in tool_calls]
    call_details = [
        {
            "name": call["name"],
            "arguments": call.get("arguments", {}),
            "ok": bool(call.get("ok", False)),
        }
        for call in tool_calls
    ]
    expected = list(case.expected_tools)
    invalid = any(not call.get("ok", False) for call in tool_calls) or bool(
        getattr(response, "errors", [])
    )
    if expected:
        first_tool_ok = bool(called) and called[0] == expected[0]
    else:
        first_tool_ok = not called
    exact_set_ok = set(called) == set(expected)
    missed = bool(expected) and not called
    false_positive = not expected and bool(called)
    return CaseResult(
        id=case.id,
        category=case.category,
        prompt=case.prompt,
        expected=expected,
        called=called,
        calls=call_details,
        first_tool_ok=first_tool_ok,
        exact_set_ok=exact_set_ok,
        invalid_call=invalid,
        missed_tool=missed,
        false_positive=false_positive,
        text=getattr(response, "text", "") or "",
        errors=list(getattr(response, "errors", []) or []),
        iterations=int(getattr(response, "iterations", 0)),
        tokens_generated=int(getattr(response, "tokens_generated", 0)),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarise(results: Sequence[CaseResult]) -> Dict[str, Any]:
    total = len(results)
    first = sum(result.first_tool_ok for result in results)
    exact = sum(result.exact_set_ok for result in results)
    invalid = sum(result.invalid_call for result in results)
    missed = sum(result.missed_tool for result in results)
    false_positive = sum(result.false_positive for result in results)
    needed = sum(1 for result in results if result.expected)
    direct = sum(1 for result in results if not result.expected)
    per_category: Dict[str, Dict[str, Any]] = {}
    for result in results:
        bucket = per_category.setdefault(
            result.category, {"cases": 0, "first_tool_ok": 0, "exact_set_ok": 0, "invalid": 0}
        )
        bucket["cases"] += 1
        bucket["first_tool_ok"] += int(result.first_tool_ok)
        bucket["exact_set_ok"] += int(result.exact_set_ok)
        bucket["invalid"] += int(result.invalid_call)
    for bucket in per_category.values():
        bucket["first_tool_accuracy"] = _ratio(bucket["first_tool_ok"], bucket["cases"])
        bucket["exact_set_accuracy"] = _ratio(bucket["exact_set_ok"], bucket["cases"])
        bucket["invalid_rate"] = _ratio(bucket["invalid"], bucket["cases"])
    return {
        "cases": total,
        "first_tool_accuracy": _ratio(first, total),
        "exact_set_accuracy": _ratio(exact, total),
        "invalid_call_rate": _ratio(invalid, total),
        "missing_tool_rate": _ratio(missed, needed),
        "no_tool_false_positive_rate": _ratio(false_positive, direct),
        "tool_use_cases": needed,
        "direct_cases": direct,
        "per_category": dict(sorted(per_category.items())),
    }


def evaluate_tool_selection(
    agent: Any,
    cases: Sequence[EvalCase],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the agent on every case and return per-case results plus a summary."""
    results: List[CaseResult] = []
    started = time.perf_counter()
    for case in cases:
        case_started = time.perf_counter()
        try:
            response = agent.chat(case.prompt)
        except Exception as exc:  # a broken case must not abort the suite
            results.append(
                CaseResult(
                    id=case.id,
                    category=case.category,
                    prompt=case.prompt,
                    expected=list(case.expected_tools),
                    called=[],
                    calls=[],
                    first_tool_ok=False,
                    exact_set_ok=False,
                    invalid_call=True,
                    missed_tool=bool(case.expected_tools),
                    false_positive=False,
                    text="",
                    errors=[f"{type(exc).__name__}: {exc}"],
                    iterations=0,
                    tokens_generated=0,
                    elapsed_sec=time.perf_counter() - case_started,
                )
            )
            continue
        result = _score_case(case, response)
        result.elapsed_sec = time.perf_counter() - case_started
        results.append(result)
        if verbose:
            mark = "PASS" if result.exact_set_ok else "FAIL"
            print(
                f"[{mark}] {result.id}: expected {result.expected or 'no tool'}, "
                f"called {result.called or 'none'} "
                f"({'invalid' if result.invalid_call else 'valid'})"
            )
    summary = summarise(results)
    summary["wall_clock_sec"] = time.perf_counter() - started
    summary["cases_with_parse_errors"] = sum(1 for result in results if result.errors)
    return {
        "summary": summary,
        "results": [result.to_dict() for result in results],
    }


def format_summary(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"cases: {summary['cases']} "
        f"({summary['tool_use_cases']} tool, {summary['direct_cases']} direct)",
        f"first-tool accuracy:  {summary['first_tool_accuracy']:.1%}",
        f"exact-set accuracy:   {summary['exact_set_accuracy']:.1%}",
        f"invalid-call rate:    {summary['invalid_call_rate']:.1%}",
        f"missing-tool rate:    {summary['missing_tool_rate']:.1%}",
        f"no-tool false pos.:   {summary['no_tool_false_positive_rate']:.1%}",
        f"parse errors:         {summary.get('cases_with_parse_errors', 0)} case(s)",
        f"wall clock:           {summary.get('wall_clock_sec', 0.0):.1f}s",
        "",
        "per category:",
    ]
    for category, bucket in summary["per_category"].items():
        lines.append(
            f"  {category:<16} n={bucket['cases']:<3} "
            f"first={bucket['first_tool_accuracy']:.0%} "
            f"exact={bucket['exact_set_accuracy']:.0%} "
            f"invalid={bucket['invalid_rate']:.0%}"
        )
    return "\n".join(lines)
