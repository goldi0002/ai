from __future__ import annotations

import time

import pytest

from tools.base import Tool, ToolResult
from tools.calculator import evaluate
from tools.files import ListFiles, ReadDocument
from tools.registry import ToolRegistry


def test_calculator_exact_results():
    assert evaluate("17*23+4") == 395
    assert evaluate("(2 + 3) * 4") == 20
    assert evaluate("sqrt(81)") == 9
    assert evaluate("2 ** 10") == 1024
    assert evaluate("round(pi, 4)") == 3.1416
    assert evaluate("max(3, 8, 2)") == 8


def test_calculator_rejects_code_and_bad_input():
    for expression in [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "x + 1",
        "2 ** 1000",
        "10 / 0",
        "1" * 300,
        "",
        "lambda: 1",
    ]:
        with pytest.raises(Exception):
            evaluate(expression)


def test_registry_lists_schemas():
    registry = ToolRegistry([ReadDocument("datasets/documents"), ListFiles("datasets/documents")])
    names = registry.names()
    assert names == ["list_files", "read_document"]
    schemas = registry.schemas()
    assert schemas[0]["function"]["name"] == "list_files"
    assert "parameters" in schemas[0]["function"]


def test_registry_unknown_tool():
    registry = ToolRegistry([ReadDocument("datasets/documents")])
    result = registry.execute("nope", {})
    assert result.ok is False
    assert "unknown tool" in result.error


def test_registry_invalid_json_arguments():
    registry = ToolRegistry([ReadDocument("datasets/documents")])
    result = registry.execute("read_document", "{not json")
    assert result.ok is False
    assert "not valid JSON" in result.error


def test_registry_missing_and_unknown_arguments():
    registry = ToolRegistry([ReadDocument("datasets/documents")])
    missing = registry.execute("read_document", {})
    assert missing.ok is False and "missing required" in missing.error
    unknown = registry.execute("read_document", {"path": "x.md", "colour": "red"})
    assert unknown.ok is False and "unknown argument" in unknown.error
    wrong_type = registry.execute("read_document", {"path": 5})
    assert wrong_type.ok is False and "must be a string" in wrong_type.error


def test_read_document_and_list_files():
    reader = ReadDocument("datasets/documents")
    result = reader.run("quantization.md")
    assert result.ok and "quantisation" in result.content.lower()
    listing = ListFiles("datasets/documents").run("*.md")
    assert listing.ok and "bm25_and_fts5.md" in listing.content


def test_read_document_rejects_escape_and_missing():
    reader = ReadDocument("datasets/documents")
    escaped = reader.run("../configs/tiny50.yaml")
    assert escaped.ok is False and "escapes" in escaped.error
    missing = reader.run("does_not_exist.md")
    assert missing.ok is False and "not found" in missing.error


def test_list_files_rejects_parent_glob():
    result = ListFiles("datasets/documents").run("../*.md")
    assert result.ok is False


class _SlowTool(Tool):
    name = "slow"
    description = "sleeps"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        time.sleep(0.3)
        return ToolResult(ok=True, content="done")


def test_registry_timeout():
    registry = ToolRegistry([_SlowTool()])
    started = time.perf_counter()
    result = registry.execute("slow", {}, timeout=0.05)
    elapsed = time.perf_counter() - started
    assert result.ok is False and "timed out" in result.error
    assert elapsed < 0.3


class _BoomTool(Tool):
    name = "boom"
    description = "raises"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        raise RuntimeError("intentional failure")


def test_tool_exception_becomes_failure():
    result = ToolRegistry([_BoomTool()]).execute("boom", {})
    assert result.ok is False
    assert "intentional failure" in result.error
