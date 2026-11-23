"""Offline research engine: SQLite + FTS5 with BM25 ranking.

Documents are chunked, stored in SQLite, and indexed with an FTS5 external
content table. Search returns chunks with document path, title, chunk ordinal
and character offsets so every answer can cite a real, inspectable passage.

There is no network access anywhere in this module.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .base import ToolError
from .files import extract_title, read_text_file

SCHEMA_VERSION = 1
DEFAULT_DB = "storage/tinylm.db"
MAX_CHUNK_CHARS = 4000
MIN_CHUNK_CHARS = 20


@dataclass
class SearchHit:
    chunk_id: int
    document_id: int
    path: str
    title: str
    seq: int
    text: str
    start_char: int
    score: float

    def citation(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "path": self.path,
            "chunk": self.seq,
            "start_char": self.start_char,
            "end_char": self.start_char + len(self.text),
        }

    def citation_text(self) -> str:
        return f"{self.title} ({self.path}, chunk {self.seq})"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["citation"] = self.citation_text()
        return payload


def chunk_text(text: str, chunk_chars: int = 800, overlap: int = 120) -> List[tuple[int, str]]:
    """Split text on paragraph/sentence boundaries with character overlap.

    Returns ``[(start_char, chunk), ...]``. The window is moved back by
    ``overlap`` characters so a passage spanning a boundary is retrievable.
    """
    if chunk_chars < 100:
        raise ValueError("chunk_chars must be >= 100")
    overlap = max(0, min(overlap, chunk_chars // 2))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: List[tuple[int, str]] = []
    position = 0
    length = len(text)
    while position < length:
        end = min(length, position + chunk_chars)
        if end < length:
            window_start = position + chunk_chars // 2
            best = -1
            for separator in ("\n\n", "\n", ". ", " "):
                found = text.rfind(separator, window_start, end)
                if found > best:
                    best = found + len(separator)
            if best > position:
                end = best
        piece = text[position:end]
        stripped = piece.strip()
        if len(stripped) >= MIN_CHUNK_CHARS:
            leading = len(piece) - len(piece.lstrip())
            chunks.append((position + leading, stripped))
        if end >= length:
            break
        position = max(position + 1, end - overlap)
    return chunks


def build_match_query(query: str, operator: str = "AND") -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Terms are quoted, so FTS5 operators and syntax cannot leak in from the
    user's string. Quoted phrases in the query are preserved as phrases.
    """
    query = (query or "").strip()
    if not query:
        return ""
    phrases = re.findall(r'"([^"]+)"', query)
    remainder = re.sub(r'"[^"]+"', " ", query)
    terms = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_\-]*", remainder)

    def quote(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    parts = [quote(phrase.strip()) for phrase in phrases if phrase.strip()]
    parts += [quote(term) for term in terms if term]
    if not parts:
        return ""
    return f" {operator} ".join(parts)


class ResearchEngine:
    """Document ingestion and BM25 search backed by SQLite FTS5."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        chunk_chars: int = 800,
        overlap: int = 120,
        root: str | Path = "datasets/documents",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(root)
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self._init_schema()

    # ------------------------------------------------------------------- db
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
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    chars INTEGER NOT NULL,
                    chunks INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, seq);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='id',
                    tokenize='porter unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES ('delete', old.id, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES ('delete', old.id, old.text);
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END;
                """
            )
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # --------------------------------------------------------------- ingest
    def ingest_text(
        self,
        text: str,
        path: str,
        title: Optional[str] = None,
        replace: bool = True,
    ) -> Dict[str, Any]:
        path = str(Path(path).as_posix())
        if not text.strip():
            raise ToolError(f"document {path} is empty")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunks = chunk_text(text, self.chunk_chars, self.overlap)
        if not chunks:
            raise ToolError(f"document {path} produced no usable chunks")
        title = title or extract_title(text, Path(path).stem)
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            existing = db.execute(
                "SELECT id, sha256 FROM documents WHERE path = ?", (path,)
            ).fetchone()
            if existing and existing["sha256"] == digest and not replace:
                return {
                    "path": path,
                    "status": "unchanged",
                    "chunks": len(chunks),
                    "document_id": existing["id"],
                }
            if existing:
                document_id = existing["id"]
                db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                db.execute(
                    "UPDATE documents SET title=?, sha256=?, chars=?, chunks=?, ingested_at=? WHERE id=?",
                    (title, digest, len(text), len(chunks), now, document_id),
                )
                status = "replaced"
            else:
                cursor = db.execute(
                    "INSERT INTO documents(path, title, sha256, chars, chunks, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (path, title, digest, len(text), len(chunks), now),
                )
                document_id = int(cursor.lastrowid)
                status = "added"
            db.executemany(
                "INSERT INTO chunks(document_id, seq, start_char, text) VALUES (?, ?, ?, ?)",
                [(document_id, seq, start, chunk) for seq, (start, chunk) in enumerate(chunks)],
            )
        return {
            "path": path,
            "title": title,
            "status": status,
            "document_id": document_id,
            "chunks": len(chunks),
            "chars": len(text),
        }

    def ingest_file(self, path: str | Path, replace: bool = True) -> Dict[str, Any]:
        """Ingest a file; ``path`` may be absolute or relative to the root."""
        file_path = Path(path)
        if not file_path.is_absolute():
            candidate = self.root / file_path
            file_path = candidate if candidate.exists() else file_path
        if not file_path.exists():
            raise ToolError(f"file not found: {path}")
        text = read_text_file(file_path)
        try:
            relative = file_path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            relative = file_path.as_posix()
        return self.ingest_text(text, relative, replace=replace)

    def ingest_directory(
        self,
        root: str | Path | None = None,
        pattern: str = "**/*",
        replace: bool = True,
    ) -> Dict[str, Any]:
        root_path = Path(root) if root else self.root
        if not root_path.exists():
            raise ToolError(f"directory does not exist: {root_path}")
        added, replaced, unchanged, failed = [], [], [], []
        for path in sorted(root_path.glob(pattern)):
            if not path.is_file() or path.suffix.lower() not in {
                ".txt", ".md", ".markdown", ".pdf", ".rst", ".py", ".json", ".csv"
            }:
                continue
            try:
                result = self.ingest_file(path, replace=replace)
            except Exception as exc:  # report per-file failures, keep going
                failed.append({"path": str(path), "error": str(exc)})
                continue
            {"added": added, "replaced": replaced, "unchanged": unchanged}[result["status"]].append(
                result["path"]
            )
        return {
            "root": str(root_path),
            "added": added,
            "replaced": replaced,
            "unchanged": unchanged,
            "failed": failed,
            "documents": len(added) + len(replaced) + len(unchanged),
        }

    # --------------------------------------------------------------- search
    def search(self, query: str, limit: int = 5, min_score: float = 0.0) -> List[SearchHit]:
        limit = max(1, min(int(limit), 50))
        match = build_match_query(query, operator="AND")
        if not match:
            return []
        rows = self._run_search(match, limit)
        if not rows:
            rows = self._run_search(build_match_query(query, operator="OR"), limit)
        hits = []
        for row in rows:
            score = -float(row["rank"])
            if score < min_score:
                continue
            hits.append(
                SearchHit(
                    chunk_id=int(row["id"]),
                    document_id=int(row["document_id"]),
                    path=row["path"],
                    title=row["title"],
                    seq=int(row["seq"]),
                    text=row["text"],
                    start_char=int(row["start_char"]),
                    score=score,
                )
            )
        return hits

    def _run_search(self, match: str, limit: int) -> List[sqlite3.Row]:
        if not match:
            return []
        with self.connect() as db:
            try:
                return db.execute(
                    """
                    SELECT c.id, c.document_id, c.seq, c.start_char, c.text,
                           d.path, d.title, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN documents d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (match, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []

    # ------------------------------------------------------------------ misc
    def list_documents(self) -> List[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, path, title, chars, chunks, ingested_at FROM documents ORDER BY path"
            ).fetchall()
        return [dict(row) for row in rows]

    def remove_document(self, path: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT id FROM documents WHERE path = ?", (str(path),)).fetchone()
            if row is None:
                return False
            db.execute("DELETE FROM chunks WHERE document_id = ?", (row["id"],))
            db.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
        return True

    def stats(self) -> Dict[str, Any]:
        with self.connect() as db:
            documents = db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            characters = db.execute("SELECT COALESCE(SUM(chars), 0) AS n FROM documents").fetchone()["n"]
        return {
            "db_path": str(self.db_path),
            "documents": documents,
            "chunks": chunks,
            "characters": characters,
        }


def format_hits(hits: Sequence[SearchHit], max_chars: int = 2500) -> str:
    """Render hits as numbered, citable passages for the model context."""
    if not hits:
        return "No matching passages were found in the local research index."
    blocks = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        header = f"[{number}] {hit.title} ({hit.path}, chunk {hit.seq})"
        body = hit.text.strip()
        remaining = max_chars - used - len(header) - 4
        if remaining <= 0:
            break
        if len(body) > remaining:
            body = body[:remaining].rstrip() + " ..."
        block = f"{header}\n{body}"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)
