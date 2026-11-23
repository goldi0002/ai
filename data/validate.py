"""Conversation / instruction dataset validation.

Malformed conversations are rejected *before* training: wrong role order, empty
turns, unbalanced tool calls, invalid JSON in tool calls and oversized samples.
Instruction data is stored separately from pretraining text (see
``datasets/instructions/`` vs ``datasets/pretraining/``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

VALID_ROLES = {"system", "user", "assistant", "tool"}
MAX_TURNS = 64
MAX_CHARS = 20_000


@dataclass
class ValidationResult:
    valid: int = 0
    invalid: int = 0
    errors: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.invalid += 1
        if len(self.errors) < 50:
            self.errors.append(message)

    def summary(self) -> str:
        return f"valid={self.valid} invalid={self.invalid}"


def _validate_tool_calls(message: Dict[str, Any], where: str, result: ValidationResult) -> None:
    calls = message.get("tool_calls")
    if calls is None:
        return
    if not isinstance(calls, list):
        result.add_error(f"{where}: tool_calls must be a list")
        return
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
            result.add_error(f"{where}: tool call {index} must have name and arguments")
            continue
        if not isinstance(call["name"], str) or not call["name"]:
            result.add_error(f"{where}: tool call {index} name must be a non-empty string")
        if not isinstance(call["arguments"], dict):
            result.add_error(f"{where}: tool call {index} arguments must be an object")


def validate_conversation(
    sample: Dict[str, Any],
    index: int = 0,
    required_system: bool = False,
) -> List[str]:
    """Return a list of problems (empty list means the sample is valid)."""
    where = f"sample[{index}]"
    errors: List[str] = []
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return [f"{where}: messages must be a non-empty list"]
    if len(messages) > MAX_TURNS:
        errors.append(f"{where}: too many turns ({len(messages)} > {MAX_TURNS})")

    total_chars = 0
    for position, message in enumerate(messages):
        spot = f"{where}.messages[{position}]"
        if not isinstance(message, dict):
            errors.append(f"{spot}: message must be an object")
            continue
        role = message.get("role")
        if role not in VALID_ROLES:
            errors.append(f"{spot}: invalid role {role!r}")
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            errors.append(f"{spot}: content must be a string")
            continue
        if not content.strip() and not message.get("tool_calls"):
            errors.append(f"{spot}: empty content")
        total_chars += len(content)
        if role == "assistant":
            _validate_tool_calls(message, spot, ValidationResult(valid=0, invalid=0, errors=errors))
        if position == 0 and role != "system" and required_system:
            errors.append(f"{spot}: first message must be system")

    if total_chars > MAX_CHARS:
        errors.append(f"{where}: conversation too long ({total_chars} chars)")
    if not any(m.get("role") == "assistant" for m in messages if isinstance(m, dict)):
        errors.append(f"{where}: no assistant turn (nothing to learn from)")
    return errors


def validate_file(path: str | Path, required_system: bool = False) -> ValidationResult:
    result = ValidationResult()
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                result.add_error(f"line {line_number}: invalid JSON ({exc})")
                continue
            problems = validate_conversation(sample, index=line_number, required_system=required_system)
            if problems:
                for problem in problems:
                    result.add_error(problem)
            else:
                result.valid += 1
    return result


def load_jsonl(path: str | Path, required_system: bool = False) -> List[Dict[str, Any]]:
    """Load a JSONL conversation file, raising on the first invalid sample."""
    samples: List[Dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        problems = validate_conversation(sample, index=line_number, required_system=required_system)
        if problems:
            raise ValueError("; ".join(problems))
        samples.append(sample)
    if not samples:
        raise ValueError(f"no samples found in {path}")
    return samples


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate TinyLM conversation JSONL files")
    parser.add_argument("paths", nargs="+", help="JSONL files to validate")
    parser.add_argument("--require-system", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.paths:
        result = validate_file(path, required_system=args.require_system)
        print(f"{path}: {result.summary()}")
        for error in result.errors:
            print(f"  error: {error}")
        if result.invalid:
            exit_code = 1 if args.fail_on_error else exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
