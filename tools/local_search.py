"""Agent-facing research search tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import Tool, ToolError, ToolResult
from .research import ResearchEngine, format_hits


class LocalSearch(Tool):
    name = "local_search"
    description = (
        "Search the local offline document index with SQLite FTS5/BM25. Returns "
        "the most relevant passages with a real citation (title, path, chunk) "
        "for each result. Use this instead of guessing facts."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords or a quoted phrase to search for."},
            "limit": {
                "type": "integer",
                "description": "Maximum passages to return (default 5, max 20).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, engine: ResearchEngine) -> None:
        self.engine = engine

    def run(self, query: str, limit: int = 5, **_: Any) -> ToolResult:
        query = (query or "").strip()
        if not query:
            return ToolResult.failure("query is empty")
        try:
            hits = self.engine.search(query, limit=max(1, min(int(limit), 20)))
        except ToolError as exc:
            return ToolResult.failure(str(exc))
        if not hits:
            return ToolResult(
                ok=True,
                content="No matching passages were found in the local research index.",
                data={"query": query, "hits": []},
                citations=[],
            )
        return ToolResult(
            ok=True,
            content=format_hits(hits),
            data={"query": query, "hits": [hit.to_dict() for hit in hits]},
            citations=[hit.citation() for hit in hits],
        )
