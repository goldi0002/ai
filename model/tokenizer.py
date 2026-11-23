"""SentencePiece BPE tokenizer wrapper for TinyLM-50.

The tokenizer is trained from the curated corpus (never assumed), writes its
special-token ids to a sibling ``*_meta.json`` file, and refuses to encode into
ids the model cannot embed. Special tokens are SentencePiece user-defined
symbols, so they are atomic pieces and survive round-trips exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import sentencepiece as spm

from .config import SPECIAL_TOKENS, TinyLMConfig

# Meta tokens that SentencePiece treats specially; the rest become user-defined
# symbols with fixed ids 4..11.
_META = ["<pad>", "<unk>", "<bos>", "<eos>"]
_USER_SYMBOLS = [token for token in SPECIAL_TOKENS if token not in _META]


class TokenizerError(RuntimeError):
    pass


class TinyTokenizer:
    """Thin, validated wrapper around SentencePieceProcessor."""

    def __init__(self, model_path: str | Path, meta_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"tokenizer model not found: {self.model_path}. "
                "Run: python -m data.prepare tokenize"
            )
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.vocab_size = self.sp.get_piece_size()
        self.special_ids: Dict[str, int] = self._collect_special_ids()
        self.meta_path = Path(meta_path) if meta_path else self.model_path.with_name(
            self.model_path.stem + "_meta.json"
        )

    # ------------------------------------------------------------ internals
    def _collect_special_ids(self) -> Dict[str, int]:
        ids: Dict[str, int] = {}
        for token in SPECIAL_TOKENS:
            piece_id = self.sp.piece_to_id(token)
            if piece_id == self.sp.unk_id() and token != "<unk>":
                raise TokenizerError(f"special token {token!r} is missing from the tokenizer")
            ids[token] = int(piece_id)
        if len(set(ids.values())) != len(ids):
            raise TokenizerError(f"special token ids collide: {ids}")
        return ids

    def verify(self, vocab_size: int | None = None) -> None:
        if vocab_size is not None and self.vocab_size != vocab_size:
            raise TokenizerError(
                f"tokenizer vocabulary {self.vocab_size} != model vocabulary {vocab_size}"
            )
        for token, expected_dtype in (("<pad>", 0),):
            if self.special_ids[token] != expected_dtype:
                raise TokenizerError(f"{token} should have id 0, got {self.special_ids[token]}")

    # -------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        corpus_paths: Sequence[str | Path],
        model_prefix: str | Path,
        vocab_size: int = 16000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
        input_sentence_size: int = 2_000_000,
        seed_sentencepiece_size: int = 1_000_000,
        hard_vocab_limit: bool = True,
    ) -> "TinyTokenizer":
        """Train a SentencePiece model and return the loaded tokenizer."""
        model_prefix = Path(model_prefix)
        model_prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=[str(p) for p in corpus_paths],
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            user_defined_symbols=_USER_SYMBOLS,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            byte_fallback=True,
            split_digits=True,
            # Case-preserving NFKC: a chat model must be able to emit capitals.
            normalization_rule_name="nmt_nfkc",
            input_sentence_size=input_sentence_size,
            seed_sentencepiece_size=seed_sentencepiece_size,
            num_threads=2,
            hard_vocab_limit=hard_vocab_limit,
        )
        tokenizer = cls(f"{model_prefix}.model")
        tokenizer.save_meta(model_prefix.with_name(model_prefix.name + "_meta.json"))
        return tokenizer

    # ------------------------------------------------------------------- I/O
    def save_meta(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.meta_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_file": self.model_path.name,
            "vocab_size": self.vocab_size,
            "model_type": "bpe",
            "special_tokens": SPECIAL_TOKENS,
            "special_ids": self.special_ids,
            "unk_id": int(self.sp.unk_id()),
            "bos_id": int(self.sp.bos_id()),
            "eos_id": int(self.sp.eos_id()),
            "pad_id": int(self.sp.pad_id()),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load_meta(model_path: str | Path) -> dict:
        model_path = Path(model_path)
        meta = model_path.with_name(model_path.stem + "_meta.json")
        if not meta.exists():
            raise FileNotFoundError(f"tokenizer metadata missing: {meta}")
        return json.loads(meta.read_text(encoding="utf-8"))

    def config_ids(self) -> Dict[str, int]:
        return {
            "pad_token_id": self.special_ids["<pad>"],
            "unk_token_id": self.special_ids["<unk>"],
            "bos_token_id": self.special_ids["<bos>"],
            "eos_token_id": self.special_ids["<eos>"],
        }

    def apply_to_config(self, config: TinyLMConfig) -> TinyLMConfig:
        """Return the model config with this tokenizer's ids and vocabulary."""
        config.vocab_size = self.vocab_size
        for field, value in self.config_ids().items():
            setattr(config, field, value)
        self.verify(config.vocab_size)
        return config

    # -------------------------------------------------------------- encoding
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.special_ids["<bos>"])
        if add_eos:
            ids.append(self.special_ids["<eos>"])
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            keep = set(self.special_ids.values())
            ids = [i for i in ids if i not in keep]
        return self.sp.decode(ids)

    def id(self, token: str) -> int:
        if token not in self.special_ids:
            raise KeyError(f"unknown special token: {token}")
        return self.special_ids[token]

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"TinyTokenizer(vocab_size={self.vocab_size}, path={self.model_path})"
