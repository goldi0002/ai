"""Agent-facing memory tools backed by :class:`memory.store.MemoryStore`."""

from __future__ import annotations

from typing import Any, Dict

from memory.store import MemoryStore

from .base import Tool, ToolResult


class RetrieveMemory(Tool):
    name = "retrieve_memory"
    description = (
        "Look up durable notes the user asked you to remember. Search by keywords "
        "or return the most important notes when the query is empty."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search memory for."},
            "limit": {"type": "integer", "description": "Maximum notes to return (default 3)."},
        },
        "required": [],
    }

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, query: str = "", limit: int = 3, **_: Any) -> ToolResult:
        entries = self.store.search(query, limit) if query.strip() else self.store.list(limit)
        if not entries:
            return ToolResult(ok=True, content="No matching memories.", data={"memories": []})
        content = "\n".join(
            f"- {entry['key']}: {entry['content']} (tags: {entry['tags'] or 'none'}, "
            f"updated {entry['updated_at']})"
            for entry in entries
        )
        return ToolResult(ok=True, content=content, data={"memories": entries})


class SaveMemory(Tool):
    name = "save_memory"
    description = (
        "Save a durable note for later conversations. Saving an existing key "
        "updates it in place. Use short, specific keys such as 'user_timezone'."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Short stable identifier."},
            "content": {"type": "string", "description": "The fact or preference to remember."},
            "tags": {"type": "string", "description": "Optional comma-separated tags."},
            "importance": {
                "type": "number",
                "description": "0..1 importance used when ordering notes (default 0.5).",
            },
        },
        "required": ["key", "content"],
    }

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        key: str,
        content: str,
        tags: str = "",
        importance: float = 0.5,
        **_: Any,
    ) -> ToolResult:
        try:
            entry = self.store.save(key, content, tags=tags, importance=importance)
        except (ValueError, TypeError) as exc:
            return ToolResult.failure(str(exc))
        return ToolResult(
            ok=True,
            content=f"Memory {entry['status']}: {entry['key']}",
            data=entry,
        )
