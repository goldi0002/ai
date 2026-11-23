"""Text cleaning and quality filtering for the TinyLM-50 data pipeline.

All functions are deterministic and dependency-light so the same cleaning can be
replayed from the dataset manifest.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_URL_ONLY = re.compile(r"^https?://\S+$", re.MULTILINE)


def normalize_text(text: str) -> str:
    """NFKC normalise, strip control characters, unify newlines and spacing."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    symbols = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'\"()-"))
    return symbols / len(text)


def _max_char_run(text: str) -> int:
    best = current = 1
    for prev, ch in zip(text, text[1:]):
        current = current + 1 if ch == prev else 1
        best = max(best, current)
    return best if text else 0


@dataclass
class QualityReport:
    keep: bool
    reason: str
    chars: int
    words: int


def quality_check(
    text: str,
    min_chars: int = 200,
    min_words: int = 30,
    max_symbol_ratio: float = 0.25,
    max_char_run: int = 80,
) -> QualityReport:
    words = len(text.split())
    if len(text) < min_chars:
        return QualityReport(False, "too_short", len(text), words)
    if words < min_words:
        return QualityReport(False, "too_few_words", len(text), words)
    if _symbol_ratio(text) > max_symbol_ratio:
        return QualityReport(False, "symbol_heavy", len(text), words)
    if _max_char_run(text) > max_char_run:
        return QualityReport(False, "repetitive_run", len(text), words)
    if len(set(text.split())) < 5:
        return QualityReport(False, "low_diversity", len(text), words)
    # Boilerplate pages that are mostly navigation/URLs.
    non_url_lines = [ln for ln in text.split("\n") if not _URL_ONLY.match(ln.strip())]
    if sum(len(ln) for ln in non_url_lines) < 0.5 * len(text):
        return QualityReport(False, "url_heavy", len(text), words)
    return QualityReport(True, "ok", len(text), words)


def remove_repeated_blocks(text: str, min_block_lines: int = 3, max_repeats: int = 2) -> str:
    """Drop paragraph blocks that appear more than ``max_repeats`` times."""
    blocks = text.split("\n\n")
    seen: dict[str, int] = {}
    kept = []
    for block in blocks:
        key = " ".join(block.split()).lower()
        if len(key.split()) >= min_block_lines * 5:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > max_repeats:
                continue
        kept.append(block)
    return "\n\n".join(kept)


def clean_documents(
    documents: Iterable[tuple[str, str]],
    min_chars: int = 200,
    min_words: int = 30,
) -> tuple[list[tuple[str, str]], dict]:
    """Clean (doc_id, text) pairs and return kept documents plus statistics."""
    kept: list[tuple[str, str]] = []
    reasons: dict[str, int] = {}
    for doc_id, raw in documents:
        text = normalize_text(raw)
        text = remove_repeated_blocks(text)
        report = quality_check(text, min_chars=min_chars, min_words=min_words)
        reasons[report.reason] = reasons.get(report.reason, 0) + 1
        if report.keep:
            kept.append((doc_id, text))
    stats = {
        "input_documents": sum(reasons.values()),
        "kept_documents": len(kept),
        "dropped_documents": sum(reasons.values()) - len(kept),
        "drop_reasons": reasons,
    }
    return kept, stats
