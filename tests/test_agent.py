from __future__ import annotations

import json

from agent.agent import Agent, AgentConfig
from agent.parser import parse_tool_calls
from agent.prompts import render_prompt
from memory.store import MemoryStore
from tools.base import Tool, ToolResult
from tools.calculator import Calculator
from tools.local_search import LocalSearch
from tools.registry import ToolRegistry
from tools.research import ResearchEngine


def _scripted(tokenizer, replies):
    iterator = iter(replies)

    def generate(prompt_ids, max_new_tokens):
        try:
            text = next(iterator)
        except StopIteration as exc:  # pragma: no cover - test bug guard
            raise AssertionError("agent asked for more generations than scripted") from exc
        return tokenizer.encode(text)

    return generate


def test_parser_handles_valid_calls_and_text():
    raw = (
        '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>\n'
        "<final>The answer is 4."
    )
    parsed = parse_tool_calls(raw)
    assert parsed.errors == []
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "calculator"
    assert parsed.tool_calls[0].arguments == {"expression": "2+2"}
    assert parsed.text == "The answer is 4."


def test_parser_reports_bad_json_and_bad_arguments():
    parsed = parse_tool_calls('<tool_call>{"name": calculator, "arguments": {}}')
    assert not parsed.tool_calls
    assert parsed.errors
    parsed = parse_tool_calls('<tool_call>{"name": "calculator", "arguments": "nope"}</tool_call>')
    assert not parsed.tool_calls
    parsed = parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>')
    assert not parsed.tool_calls and "name" in parsed.errors[0]


def test_parser_drops_duplicate_calls():
    raw = (
        '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}'
        '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}'
    )
    parsed = parse_tool_calls(raw)
    assert len(parsed.tool_calls) == 1
    assert any("duplicate" in error for error in parsed.errors)


def test_render_prompt_respects_budget(tokenizer):
    turns = [("user", "word " * 500), ("user", "What is 2+2?")]
    prompt = render_prompt(tokenizer, "You are a test.", turns, max_tokens=200)
    assert prompt.startswith("<system>")
    assert prompt.endswith("<assistant>")
    assert len(tokenizer.encode(prompt)) <= 260


def test_agent_executes_tool_and_returns_final(tokenizer):
    registry = ToolRegistry([Calculator()])
    replies = [
        '<tool_call>{"name": "calculator", "arguments": {"expression": "17*23+4"}}',
        "<final>17*23+4 = 395.",
    ]
    agent = Agent(
        None,
        tokenizer,
        registry,
        config=AgentConfig(max_iterations=3, max_context_tokens=512),
        generate_fn=_scripted(tokenizer, replies),
    )
    response = agent.chat("What is 17*23+4?")
    assert response.text == "17*23+4 = 395."
    assert response.stopped_reason == "final"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["ok"] is True
    assert "395" in response.tool_calls[0]["content"]
    assert response.iterations == 2


def test_agent_direct_answer_without_tools(tokenizer):
    agent = Agent(
        None,
        tokenizer,
        ToolRegistry([Calculator()]),
        config=AgentConfig(max_context_tokens=512),
        generate_fn=_scripted(tokenizer, ["<final>Hello there."]),
    )
    response = agent.chat("hi")
    assert response.text == "Hello there."
    assert response.tool_calls == []
    assert response.iterations == 1


def test_agent_unknown_tool_is_reported(tokenizer):
    registry = ToolRegistry([Calculator()])
    replies = [
        '<tool_call>{"name": "launch_rocket", "arguments": {}}',
        "<final>I do not have that tool.",
    ]
    agent = Agent(
        None,
        tokenizer,
        registry,
        config=AgentConfig(max_context_tokens=512),
        generate_fn=_scripted(tokenizer, replies),
    )
    response = agent.chat("launch a rocket")
    assert response.tool_calls[0]["ok"] is False
    assert "unknown tool" in response.tool_calls[0]["error"]
    assert response.text == "I do not have that tool."


def test_agent_retries_after_malformed_tool_json(tokenizer):
    replies = [
        '<tool_call>{"name": "calculator", "arguments": {"expression": ',  # truncated
        '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}',
        "<final>2+2 = 4.",
    ]
    agent = Agent(
        None,
        tokenizer,
        ToolRegistry([Calculator()]),
        config=AgentConfig(max_iterations=3, max_context_tokens=512),
        generate_fn=_scripted(tokenizer, replies),
    )
    response = agent.chat("2+2?")
    assert response.tool_calls and response.tool_calls[0]["ok"]
    assert response.text == "2+2 = 4."
    assert response.iterations == 3


def test_agent_iteration_limit(tokenizer):
    registry = ToolRegistry([Calculator()])
    agent = Agent(
        None,
        tokenizer,
        registry,
        config=AgentConfig(max_iterations=1, max_context_tokens=512),
        generate_fn=_scripted(
            tokenizer, ['<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}']
        ),
    )
    response = agent.chat("1+1?")
    assert response.stopped_reason == "max_iterations"
    assert response.iterations == 1


def test_agent_repeated_tool_call_guard(tokenizer):
    call = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}'
    agent = Agent(
        None,
        tokenizer,
        ToolRegistry([Calculator()]),
        config=AgentConfig(max_iterations=4, max_context_tokens=512),
        generate_fn=_scripted(tokenizer, [call, call]),
    )
    response = agent.chat("2+2?")
    assert response.stopped_reason == "repeated_tool_call"
    assert any("repeated" in error for error in response.errors)


def test_agent_collects_citations_from_search(tokenizer, tmp_path):
    engine = ResearchEngine(db_path=tmp_path / "research.db")
    engine.ingest_text(
        "BM25 ranks documents using term frequency and document length normalisation.",
        "bm25.md",
        title="BM25 notes",
    )
    registry = ToolRegistry([LocalSearch(engine)])
    replies = [
        '<tool_call>{"name": "local_search", "arguments": {"query": "BM25", "limit": 2}}',
        "<final>BM25 uses term frequency and length normalisation [1].",
    ]
    agent = Agent(
        None,
        tokenizer,
        registry,
        config=AgentConfig(max_context_tokens=512),
        generate_fn=_scripted(tokenizer, replies),
    )
    response = agent.chat("How does BM25 rank?")
    assert response.citations and response.citations[0]["path"] == "bm25.md"
    assert "[1]" in response.text


def test_agent_memory_tools(tokenizer, tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    from tools.memory_tools import RetrieveMemory, SaveMemory

    registry = ToolRegistry([SaveMemory(store), RetrieveMemory(store)])
    replies = [
        '<tool_call>{"name": "save_memory", "arguments": {"key": "editor", '
        '"content": "The user uses Neovim.", "tags": "preference"}}',
        "<final>Saved.",
        '<tool_call>{"name": "retrieve_memory", "arguments": {"query": "editor"}}',
        "<final>You use Neovim.",
    ]
    agent = Agent(
        None,
        tokenizer,
        registry,
        config=AgentConfig(max_context_tokens=512),
        generate_fn=_scripted(tokenizer, replies),
    )
    first = agent.chat("Remember that I use Neovim.")
    assert first.tool_calls[0]["ok"]
    second = agent.chat("Which editor do I use?")
    assert "Neovim" in second.tool_calls[0]["content"]
    assert store.get("editor")["content"] == "The user uses Neovim."
