"""Read-only filesystem tools confined to the configured document root."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import Tool, ToolError, ToolResult

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".py", ".json", ".csv", ".log", ".rst", ".yaml", ".yml"}
PDF_SUFFIXES = {".pdf"}
MAX_READ_CHARS = 20_000


def _resolve(root: Path, relative: str) -> Path:
    if not relative or not str(relative).strip():
        raise ToolError("path is empty")
    candidate = (root / str(relative)).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
    except OSError as exc:
        raise ToolError(f"invalid path: {exc}") from exc
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise ToolError(f"path escapes the document root: {relative}")
    return resolved


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in PDF_SUFFIXES:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ToolError("pypdf is not installed; cannot read PDF files") from exc
        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            pages.append(f"[page {number}]\n{page.extract_text() or ''}")
        return "\n\n".join(pages)
    raise ToolError(f"unsupported file type {suffix or '(none)'}")


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:10]:
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:120] or fallback
    return fallback


class ReadDocument(Tool):
    name = "read_document"
    description = (
        "Read a document (text, Markdown, Python source or PDF) from the local "
        "document library and return its contents."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the document root."},
            "max_chars": {
                "type": "integer",
                "description": f"Characters to return (default and maximum {MAX_READ_CHARS}).",
            },
        },
        "required": ["path"],
    }

    def __init__(self, root: str | Path = "datasets/documents") -> None:
        self.root = Path(root)

    def run(self, path: str, max_chars: int = MAX_READ_CHARS, **_: Any) -> ToolResult:
        try:
            resolved = _resolve(self.root, path)
            if not resolved.exists():
                raise ToolError(f"file not found: {path}")
            if not resolved.is_file():
                raise ToolError(f"not a file: {path}")
            text = read_text_file(resolved)
        except ToolError as exc:
            return ToolResult.failure(str(exc))
        limit = max(0, min(int(max_chars), MAX_READ_CHARS))
        truncated = len(text) > limit
        content = text[:limit]
        if truncated:
            content += f"\n\n[truncated: {len(text) - limit} more characters]"
        return ToolResult(
            ok=True,
            content=content,
            data={
                "path": str(resolved.relative_to(self.root.resolve())),
                "chars": len(text),
                "truncated": truncated,
            },
        )


class ListFiles(Tool):
    name = "list_files"
    description = "List documents available in the local library, optionally filtered by glob."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob such as '*.md' (default '**/*')."},
            "max_results": {"type": "integer", "description": "Maximum entries to return (default 100)."},
        },
        "required": [],
    }

    def __init__(self, root: str | Path = "datasets/documents") -> None:
        self.root = Path(root)

    def run(self, pattern: str = "**/*", max_results: int = 100, **_: Any) -> ToolResult:
        try:
            if not self.root.exists():
                raise ToolError(f"document root does not exist: {self.root}")
            if ".." in Path(pattern).parts:
                raise ToolError("pattern must not contain '..'")
            limit = max(1, min(int(max_results), 500))
            files: List[Dict[str, Any]] = []
            for path in sorted(self.root.glob(pattern)):
                if path.is_file() and not any(part.startswith(".") for part in path.parts):
                    files.append(
                        {
                            "path": str(path.relative_to(self.root)),
                            "bytes": path.stat().st_size,
                        }
                    )
                    if len(files) >= limit:
                        break
        except ToolError as exc:
            return ToolResult.failure(str(exc))
        if not files:
            return ToolResult(ok=True, content="No matching documents.", data={"files": []})
        listing = "\n".join(f"{item['path']} ({item['bytes']:,} bytes)" for item in files)
        return ToolResult(
            ok=True,
            content=f"{len(files)} document(s):\n{listing}",
            data={"files": files},
        )
