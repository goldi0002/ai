"""Exact and near-duplicate detection for corpus preparation.

Near-duplicate detection uses 64-bit SimHash over word 5-shingles with banded
lookup, which is O(n) in the corpus size and needs no external dependency.
Exact duplicates are caught by a SHA-256 of the normalised text first.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

_SHINGLE = 5
_BANDS = 4
_BAND_BITS = 16


def _shingles(text: str, size: int = _SHINGLE) -> List[str]:
    words = text.lower().split()
    if len(words) < size:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


def _hash64(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def simhash(text: str) -> int:
    weights = [0] * 64
    for shingle in _shingles(text):
        h = _hash64(shingle)
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class Deduplicator:
    threshold: int = 3  # max Hamming distance on 64-bit SimHash
    _buckets: Dict[Tuple[int, int], List[Tuple[int, int]]] = field(default_factory=dict)
    _exact: Dict[str, int] = field(default_factory=dict)
    kept: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0

    def _bands(self, signature: int) -> Iterable[Tuple[int, int]]:
        mask = (1 << _BAND_BITS) - 1
        for band in range(_BANDS):
            yield band, (signature >> (band * _BAND_BITS)) & mask

    def is_duplicate(self, doc_id: str, text: str) -> Tuple[bool, str | None]:
        digest = hashlib.sha256(" ".join(text.split()).lower().encode("utf-8")).hexdigest()
        if digest in self._exact:
            self.exact_duplicates += 1
            return True, "exact"
        signature = simhash(text)
        for key in self._bands(signature):
            for other_sig, _other_id in self._buckets.get(key, ()):
                if hamming(signature, other_sig) <= self.threshold:
                    self.near_duplicates += 1
                    return True, "near"
        self._exact[digest] = doc_id
        for key in self._bands(signature):
            self._buckets.setdefault(key, []).append((signature, len(self._exact)))
        self.kept += 1
        return False, None

    def stats(self) -> dict:
        return {
            "kept": self.kept,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "threshold_hamming": self.threshold,
            "shingle_size": _SHINGLE,
        }


def deduplicate(documents: Iterable[tuple[str, str]], threshold: int = 3) -> tuple[List[tuple[str, str]], dict]:
    dedup = Deduplicator(threshold=threshold)
    kept: List[tuple[str, str]] = []
    for doc_id, text in documents:
        duplicate, _ = dedup.is_duplicate(doc_id, text)
        if not duplicate:
            kept.append((doc_id, text))
    return kept, dedup.stats()
