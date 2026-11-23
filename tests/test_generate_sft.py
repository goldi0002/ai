"""The generated SFT set must be valid, deterministic and disjoint from the eval."""

import json
from pathlib import Path

import pytest

from agent.factory import build_registry
from data.generate_sft import generate
from evaluation.tool_selection import load_cases

SCALE = 0.1


@pytest.fixture(scope="module")
def generated():
    records, counts = generate(seed=7, scale=SCALE, include_hand_written=False)
    return records, counts


def test_generation_is_deterministic(generated):
    again, _ = generate(seed=7, scale=SCALE, include_hand_written=False)
    assert json.dumps(again) == json.dumps(generated[0])


def test_every_category_present(generated):
    _, counts = generated
    for category in (
        "calculator", "local_search", "read_document", "list_files", "save_memory",
        "retrieve_memory", "no_tool", "unavailable_tool", "multi_tool",
    ):
        assert counts.get(category, 0) > 0, category


def test_tool_calls_are_valid_and_results_match_real_tools(generated, tmp_path):
    records, _ = generated
    registry, engine, _ = build_registry(db_path=tmp_path / "t.db")
    engine.ingest_directory("datasets/documents")
    checked = 0
    for record in records:
        messages = record["messages"]
        for index, message in enumerate(messages):
            for tool_call in message.get("tool_calls") or []:
                registry.validate_arguments(tool_call["name"], tool_call["arguments"])
        # Calculator results are deterministic: re-running must reproduce them.
        for index, message in enumerate(messages):
            calls = message.get("tool_calls") or []
            if calls and all(c["name"] == "calculator" for c in calls):
                for offset, tool_call in enumerate(calls, start=1):
                    if index + offset < len(messages) and messages[index + offset]["role"] == "tool":
                        expected = registry.execute("calculator", tool_call["arguments"]).to_json()
                        assert messages[index + offset]["content"] == expected
                        checked += 1
    assert checked > 0


def test_no_prompt_overlaps_the_held_out_eval(generated):
    records, _ = generated
    held_out = [" ".join(c.prompt.lower().split()) for c in load_cases()]
    for record in records:
        for message in record["messages"]:
            if message["role"] != "user":
                continue
            prompt = " ".join(message["content"].lower().split())
            assert not any(h in prompt or prompt in h for h in held_out), message["content"]


def test_conversations_end_in_a_final_or_a_tool_call(generated):
    records, _ = generated
    for record in records:
        last = record["messages"][-1]
        assert last["role"] == "assistant"
        assert last["content"].startswith("<final>") or last.get("tool_calls")
