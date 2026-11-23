"""Parse model output into text and tool calls.

Expected model format::

    <tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}
    <tool_call>{"name": "local_search", "arguments": {"query": "bm25"}}
    <final>The answer is 4 [1].

Parsing is defensive: malformed JSON, unknown fields, duplicate calls and
non-object arguments are reported as errors rather than raised, so the agent
can feed the error back to the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TOOL_CALL_TOKEN = "<tool_call>"
FINAL_TOKEN = "<final>"
_TOOL_TERMINATORS = ("<tool_call>", "<tool_result>", "<eos>", "<final>", "<assistant>", "<user>")
_SPECIALS = re.compile(r"</?(pad|unk|bos|eos|user|assistant|system|tool_call|tool_result|research|evidence|final)>")


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ParsedOutput:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _extract_json(blob: str) -> Optional[str]:
    start = blob.find("{")
    if start < 0:
        return None
    try:
        _, end = json.JSONDecoder().raw_decode(blob[start:])
    except json.JSONDecodeError:
        return None
    return blob[start : start + end]


def _clean_text(text: str) -> str:
    text = text.replace(FINAL_TOKEN, " ")
    text = _SPECIALS.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_tool_calls(raw: str) -> ParsedOutput:
    raw = raw or ""
    calls: List[ToolCall] = []
    errors: List[str] = []
    remainder = raw

    while True:
        index = remainder.find(TOOL_CALL_TOKEN)
        if index < 0:
            break
        after = remainder[index + len(TOOL_CALL_TOKEN) :]
        end = len(after)
        for terminator in _TOOL_TERMINATORS:
            found = after.find(terminator)
            if 0 <= found < end:
                end = found
        blob = after[:end].strip()
        remainder = after[end:] if end < len(after) else ""
        json_text = _extract_json(blob)
        if json_text is None:
            errors.append(f"tool call is not valid JSON: {blob[:120]!r}")
            continue
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            errors.append(f"tool call JSON error: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append("tool call must be a JSON object")
            continue
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            errors.append("tool call is missing a non-empty 'name'")
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                errors.append(f"arguments for {name} are a string but not valid JSON")
                continue
        if not isinstance(arguments, dict):
            errors.append(f"arguments for {name} must be a JSON object")
            continue
        if any(call.name == name and call.arguments == arguments for call in calls):
            errors.append(f"duplicate tool call for {name} in one response")
            continue
        calls.append(ToolCall(name=name.strip(), arguments=arguments, raw=json_text))

    final_index = raw.rfind(FINAL_TOKEN)
    last_call_index = raw.rfind(TOOL_CALL_TOKEN)
    visible = raw[final_index + len(FINAL_TOKEN) :] if final_index >= 0 else raw
    text = _clean_text(visible)
    if calls and final_index < last_call_index:
        # Text around tool calls is scratchpad reasoning, not the answer.
        text = ""
    return ParsedOutput(text=text, tool_calls=calls, errors=errors, raw=raw)


def strip_special_tokens(text: str) -> str:
    return _clean_text(text)
