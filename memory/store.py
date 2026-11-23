"""Persistent memory: keyed notes with FTS5 retrieval.

Memory is user-visible and editable. Each entry has a stable key, so saving the
same key again updates the note instead of duplicating it, and retrieval
returns the stored text with its update time so the agent never silently
rewrites a fact.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DB = "storage/tinylm.db"
MAX_CONTENT_CHARS = 4000
MAX_KEY_CHARS = 120


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _match_query(query: str) -> str:
    terms = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_\-]*", query or "")
    if not terms:
        return ""
    return " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)


class MemoryStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    key, content, tags,
                    content='memories',
                    content_rowid='id',
                    tokenize='porter unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, key, content, tags)
                    VALUES (new.id, new.key, new.content, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, key, content, tags)
                    VALUES ('delete', old.id, old.key, old.content, old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, key, content, tags)
                    VALUES ('delete', old.id, old.key, old.content, old.tags);
                    INSERT INTO memories_fts(rowid, key, content, tags)
                    VALUES (new.id, new.key, new.content, new.tags);
                END;
                """
            )

    # ---------------------------------------------------------------- write
    def save(
        self,
        key: str,
        content: str,
        tags: str = "",
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        key = (key or "").strip()
        content = (content or "").strip()
        if not key:
            raise ValueError("memory key is empty")
        if len(key) > MAX_KEY_CHARS:
            raise ValueError(f"memory key longer than {MAX_KEY_CHARS} characters")
        if not content:
            raise ValueError("memory content is empty")
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(f"memory content longer than {MAX_CONTENT_CHARS} characters")
        importance = max(0.0, min(1.0, float(importance)))
        tags = ",".join(part.strip() for part in str(tags).split(",") if part.strip())
        now = _now()
        with self.connect() as db:
            existing = db.execute("SELECT id FROM memories WHERE key = ?", (key,)).fetchone()
            if existing:
                db.execute(
                    "UPDATE memories SET content=?, tags=?, importance=?, updated_at=? WHERE key=?",
                    (content, tags, importance, now, key),
                )
                status = "updated"
            else:
                db.execute(
                    "INSERT INTO memories(key, content, tags, importance, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, content, tags, importance, now, now),
                )
                status = "created"
            row = db.execute("SELECT * FROM memories WHERE key = ?", (key,)).fetchone()
        payload = dict(row)
        payload["status"] = status
        return payload

    def update(
        self,
        key: str,
        content: Optional[str] = None,
        tags: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> Dict[str, Any]:
        existing = self.get(key)
        if existing is None:
            raise KeyError(f"no memory with key {key!r}")
        return self.save(
            key,
            content if content is not None else existing["content"],
            tags if tags is not None else existing["tags"],
            importance if importance is not None else existing["importance"],
        )

    def delete(self, key: str) -> bool:
        with self.connect() as db:
            cursor = db.execute("DELETE FROM memories WHERE key = ?", (key,))
            return cursor.rowcount > 0

    # ----------------------------------------------------------------- read
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM memories WHERE key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM memories ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        match = _match_query(query)
        if not match:
            return self.list(limit)
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT m.*, bm25(memories_fts) AS rank
                FROM memories_fts
                JOIN memories m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, max(1, min(int(limit), 50))),
            ).fetchall()
        return [dict(row) for row in rows]

    def to_context(self, query: str = "", limit: int = 3, max_chars: int = 800) -> str:
        """Format memories for prompt injection; empty string when none match."""
        entries = self.search(query, limit) if query else self.list(limit)
        if not entries:
            return ""
        lines = []
        used = 0
        for entry in entries:
            line = f"- {entry['key']}: {entry['content']} (updated {entry['updated_at']})"
            if used + len(line) > max_chars:
                remaining = max_chars - used
                if remaining > 40:
                    lines.append(line[:remaining].rstrip() + " ...")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        with self.connect() as db:
            count = db.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        return {"db_path": str(self.db_path), "memories": count}
