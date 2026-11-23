"""Tool primitives shared by every TinyLM tool.

A tool has a name, a natural-language description, a JSON-schema parameter
block and a ``run`` method returning a :class:`ToolResult`. Failures are
returned as ``ok=False`` results rather than raised through the agent loop, so
the model can read the error and correct itself.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class ToolError(Exception):
    """Raised inside a tool for expected, reportable failures."""


@dataclass
class ToolResult:
    ok: bool
    content: str
    data: Any = None
    error: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        payload = {"ok": self.ok, "content": self.content}
        if self.error:
            payload["error"] = self.error
        if self.citations:
            payload["citations"] = self.citations
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def failure(cls, message: str, **kwargs: Any) -> "ToolResult":
        return cls(ok=False, content=f"Error: {message}", error=message, **kwargs)


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"
