"""Sequence packing: tokenize documents once and store a flat uint16 token file.

Layout
------
``<name>.bin``      flat little-endian uint16 token ids (vocab 16000 < 65536)
``<name>.idx.json`` metadata: sequence length, document count, byte offsets

Every document is encoded as ``<bos> ... <eos>``. Documents are concatenated and
then sliced into fixed-length training sequences; the first token of each
sequence is a real token (not padding), so no padding masks are needed for
pretraining. A trailing partial sequence is dropped from training but kept in
the file so validation can still read the last tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np

from model.tokenizer import TinyTokenizer


def _write_tokens(path: Path, ids: List[int]) -> None:
    array = np.asarray(ids, dtype=np.uint16)
    tmp = path.with_suffix(path.suffix + ".tmp")
    array.tofile(tmp)
    tmp.replace(path)


@dataclass
class PackedDataset:
    """Memory-mapped access to a packed token file."""

    bin_path: Path
    seq_len: int
    num_sequences: int
    num_tokens: int

    @classmethod
    def load(cls, bin_path: str | Path) -> "PackedDataset":
        bin_path = Path(bin_path)
        meta = json.loads(bin_path.with_suffix(".idx.json").read_text(encoding="utf-8"))
        return cls(
            bin_path=bin_path,
            seq_len=int(meta["seq_len"]),
            num_sequences=int(meta["num_sequences"]),
            num_tokens=int(meta["num_tokens"]),
        )

    def _memmap(self) -> np.ndarray:
        return np.memmap(self.bin_path, dtype=np.uint16, mode="r")

    def sequence(self, index: int) -> np.ndarray:
        data = self._memmap()
        start = index * self.seq_len
        return np.asarray(data[start : start + self.seq_len], dtype=np.int64)

    def batch(self, indices: Iterable[int]) -> np.ndarray:
        return np.stack([self.sequence(i) for i in indices])

    def __len__(self) -> int:
        return self.num_sequences

    def iter_sequences(self, batch_size: int = 8, drop_last: bool = True) -> Iterator[np.ndarray]:
        total = len(self)
        if drop_last:
            total -= total % batch_size
        for start in range(0, total, batch_size):
            indices = range(start, min(start + batch_size, len(self)))
            yield self.batch(indices)


def pack_documents(
    documents: Iterable[Tuple[str, str]],
    tokenizer: TinyTokenizer,
    out_prefix: str | Path,
    seq_len: int = 1024,
    max_documents: int | None = None,
) -> PackedDataset:
    """Tokenize and pack documents into ``.bin`` + ``.idx.json``."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ids: List[int] = []
    doc_count = 0
    for _, text in documents:
        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        ids.extend(encoded)
        doc_count += 1
        if max_documents is not None and doc_count >= max_documents:
            break
    if not ids:
        raise ValueError("no tokens produced; check the corpus before packing")

    bin_path = out_prefix.with_suffix(".bin")
    _write_tokens(bin_path, ids)
    num_sequences = len(ids) // seq_len
    meta = {
        "seq_len": seq_len,
        "num_sequences": num_sequences,
        "num_tokens": len(ids),
        "num_documents": doc_count,
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer.model_path.name,
    }
    bin_path.with_suffix(".idx.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return PackedDataset(
        bin_path=bin_path, seq_len=seq_len, num_sequences=num_sequences, num_tokens=len(ids)
    )


def split_documents(
    documents: List[Tuple[str, str]],
    valid_ratio: float = 0.02,
    seed: int = 1337,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Deterministic document-level split (never splits inside a document)."""
    import random

    rng = random.Random(seed)
    shuffled = list(documents)
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    num_valid = max(1, int(len(shuffled) * valid_ratio))
    num_valid = min(num_valid, len(shuffled) - 1)
    return shuffled[num_valid:], shuffled[:num_valid]
