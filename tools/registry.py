"""Tool registry: lookup, schema export, argument validation and execution."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, Iterable, List, Optional

from .base import Tool, ToolResult

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("tool needs a name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def describe(self) -> str:
        """Compact text description for a prompt (JSON schema included)."""
        lines = []
        for name in self.names():
            tool = self._tools[name]
            properties = tool.parameters.get("properties", {})
            params = ", ".join(
                f"{key}: {value.get('type', 'any')}" for key, value in properties.items()
            )
            required = set(tool.parameters.get("required", []))
            params = ", ".join(
                f"{key}{'*' if key in required else ''}: {value.get('type', 'any')}"
                for key, value in properties.items()
            )
            lines.append(f"- {name}({params}): {tool.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------- execution
    def validate_arguments(self, name: str, arguments: Dict[str, Any]) -> Optional[str]:
        tool = self._tools.get(name)
        if tool is None:
            return f"unknown tool {name!r}; available: {', '.join(self.names())}"
        if not isinstance(arguments, dict):
            return "arguments must be a JSON object"
        properties = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            return f"missing required argument(s): {', '.join(missing)}"
        unknown = [key for key in arguments if key not in properties]
        if unknown:
            return f"unknown argument(s) for {name}: {', '.join(unknown)}"
        for key, value in arguments.items():
            expected = properties[key].get("type")
            if expected is None:
                continue
            python_type = _TYPE_MAP.get(expected)
            if python_type is None:
                continue
            if expected in ("integer", "number") and isinstance(value, bool):
                return f"argument {key!r} must be a {expected}, got boolean"
            if not isinstance(value, python_type):
                return f"argument {key!r} must be a {expected}, got {type(value).__name__}"
        return None

    def execute(
        self,
        name: str,
        arguments: Dict[str, Any] | str | None = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return ToolResult.failure(
                    f"arguments for {name} are not valid JSON: {exc.msg}"
                )
        problem = self.validate_arguments(name, arguments)  # type: ignore[arg-type]
        if problem:
            return ToolResult.failure(problem)
        tool = self._tools[name]
        if timeout and timeout > 0:
            return self._execute_with_timeout(tool, arguments, timeout)  # type: ignore[arg-type]
        return self._run(tool, arguments)  # type: ignore[arg-type]

    @staticmethod
    def _run(tool: Tool, arguments: Dict[str, Any]) -> ToolResult:
        try:
            result = tool.run(**arguments)
        except TypeError as exc:
            return ToolResult.failure(f"{tool.name} rejected its arguments: {exc}")
        except Exception as exc:  # a tool bug must not crash the agent
            return ToolResult.failure(f"{tool.name} failed: {type(exc).__name__}: {exc}")
        if not isinstance(result, ToolResult):
            return ToolResult.failure(f"{tool.name} returned {type(result).__name__}, not ToolResult")
        return result

    def _execute_with_timeout(
        self, tool: Tool, arguments: Dict[str, Any], timeout: float
    ) -> ToolResult:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{tool.name}")
        future = executor.submit(self._run, tool, arguments)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            return ToolResult.failure(f"{tool.name} timed out after {timeout:.1f}s")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
