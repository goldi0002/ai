"""Batch sources for pretraining and instruction tuning.

Next-token convention
---------------------
The model computes ``cross_entropy(logits, targets)`` at identical positions,
so the caller shifts the sequence::

    input_ids = ids[:-1]
    labels    = ids[1:]        # -100 where the token must not be learned

That removes the trivial leak where position ``i`` could copy its own token.
Packed pretraining sequences and rendered conversations both follow this rule.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch

from data.pack import PackedDataset
from model.tokenizer import TinyTokenizer


# ------------------------------------------------------------------ pretrain
class PretrainSource:
    """Shuffled fixed-length batches from packed uint16 token files."""

    def __init__(
        self,
        train: PackedDataset,
        valid: PackedDataset | None = None,
        seed: int = 1337,
    ) -> None:
        if len(train) < 1:
            raise ValueError("training split has no full sequences")
        self.train = train
        self.valid = valid
        self.seed = seed
        self.epochs = 0
        self._rng = np.random.default_rng(seed)

    def train_batches(self, batch_size: int, max_steps: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            order = self._rng.permutation(len(self.train))
            self.epochs += 1
            usable = len(order) - len(order) % batch_size
            for start in range(0, usable, batch_size):
                indices = order[start : start + batch_size]
                array = self.train.batch(indices)  # (b, seq_len)
                ids = torch.from_numpy(array.astype(np.int64))
                yield ids[:, :-1], ids[:, 1:]

    def eval_batches(
        self, batch_size: int, num_batches: int, split: str = "valid"
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset = self.valid if split == "valid" else self.train
        if dataset is None or len(dataset) == 0:
            return []
        rng = np.random.default_rng(self.seed + 991)
        count = min(num_batches, max(1, len(dataset) // batch_size))
        batches = []
        for _ in range(count):
            order = rng.choice(len(dataset), size=batch_size, replace=len(dataset) < batch_size)
            ids = torch.from_numpy(dataset.batch(order).astype(np.int64))
            batches.append((ids[:, :-1], ids[:, 1:]))
        return batches

    def metadata(self) -> Dict[str, Any]:
        return {
            "train_sequences": len(self.train),
            "valid_sequences": len(self.valid) if self.valid else 0,
            "seq_len": self.train.seq_len,
            "train_tokens": self.train.num_tokens,
        }


# ----------------------------------------------------------------------- SFT
def read_conversations(path: str | Path) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"no conversations in {path}")
    return samples


def _tool_call_text(call: Dict[str, Any]) -> str:
    return json.dumps(
        {"name": call["name"], "arguments": call.get("arguments", {})},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def render_conversation(
    tokenizer: TinyTokenizer,
    messages: Sequence[Dict[str, Any]],
) -> Tuple[List[int], List[bool]]:
    """Render a conversation to (ids, target_mask).

    Roles map to special tokens; assistant content (including tool calls) is
    the only thing trained on. Tool results are context, never targets.
    """
    ids: List[int] = []
    targets: List[bool] = []

    def emit(token_ids: Iterable[int], target: bool) -> None:
        for token_id in token_ids:
            ids.append(int(token_id))
            targets.append(target)

    emit([tokenizer.id("<bos>")], False)
    for message in messages:
        role = message["role"]
        if role == "system":
            emit([tokenizer.id("<system>")], False)
            emit(tokenizer.encode(message.get("content", "")), False)
        elif role == "user":
            emit([tokenizer.id("<user>")], False)
            emit(tokenizer.encode(message.get("content", "")), False)
        elif role == "tool":
            emit([tokenizer.id("<tool_result>")], False)
            emit(tokenizer.encode(message.get("content", "")), False)
        elif role == "assistant":
            emit([tokenizer.id("<assistant>")], True)
            content = message.get("content") or ""
            if content:
                emit(tokenizer.encode(content), True)
            for call in message.get("tool_calls") or []:
                emit([tokenizer.id("<tool_call>")], True)
                emit(tokenizer.encode(_tool_call_text(call)), True)
            emit([tokenizer.id("<eos>")], True)
        else:
            raise ValueError(f"unsupported role {role!r}")
    if not any(targets):
        raise ValueError("conversation has no assistant tokens to train on")
    return ids, targets


def conversation_to_example(
    tokenizer: TinyTokenizer,
    messages: Sequence[Dict[str, Any]],
) -> Tuple[List[int], List[int]]:
    """Return shifted (input_ids, labels) with -100 on non-assistant positions."""
    ids, target_mask = render_conversation(tokenizer, messages)
    input_ids = ids[:-1]
    labels = [ids[index + 1] if target_mask[index + 1] else -100 for index in range(len(input_ids))]
    return input_ids, labels


def pad_batch(
    examples: Sequence[Tuple[List[int], List[int]]],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids, _ in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for row, (ids, targets) in enumerate(examples):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[row, : len(targets)] = torch.tensor(targets, dtype=torch.long)
    return input_ids, labels


class ConversationSource:
    """Fixed instruction set with a deterministic train/validation split.

    The dataset is deliberately trained as repeated epochs: instruction data is
    small, and repeating it is the normal way to fit a chat format.
    """

    def __init__(
        self,
        train_samples: Sequence[Dict[str, Any]],
        valid_samples: Sequence[Dict[str, Any]],
        tokenizer: TinyTokenizer,
        max_seq_len: int = 1024,
        seed: int = 1337,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.pad_token_id = tokenizer.special_ids["<pad>"]
        self.train_examples = self._encode(train_samples, name="train")
        self.valid_examples = self._encode(valid_samples, name="valid")
        self.skipped: Dict[str, int] = {"train": 0, "valid": 0}
        if not self.train_examples:
            raise ValueError("no training conversations fit within max_seq_len")
        self._rng = random.Random(seed)
        self.epochs = 0

    def _encode(self, samples: Sequence[Dict[str, Any]], name: str) -> List[Tuple[List[int], List[int]]]:
        examples = []
        skipped = 0
        for sample in samples:
            ids, labels = conversation_to_example(self.tokenizer, sample["messages"])
            if len(ids) > self.max_seq_len:
                skipped += 1
                continue
            examples.append((ids, labels))
        if name in getattr(self, "skipped", {}):
            self.skipped[name] = skipped
        else:
            self.skipped = {"train": 0, "valid": 0}
            self.skipped[name] = skipped
        return examples

    def train_batches(self, batch_size: int, max_steps: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        order: List[int] = []
        while True:
            if len(order) < batch_size:
                epoch = list(range(len(self.train_examples)))
                self._rng.shuffle(epoch)
                order.extend(epoch)
                self.epochs += 1
            batch_indices, order = order[:batch_size], order[batch_size:]
            yield pad_batch([self.train_examples[i] for i in batch_indices], self.pad_token_id)

    def eval_batches(self, batch_size: int, num_batches: int, split: str = "valid") -> List[Tuple[torch.Tensor, torch.Tensor]]:
        examples = self.valid_examples or self.train_examples
        batches = []
        for start in range(0, min(len(examples), batch_size * num_batches), batch_size):
            chunk = examples[start : start + batch_size]
            if chunk:
                batches.append(pad_batch(chunk, self.pad_token_id))
        return batches

    def metadata(self) -> Dict[str, Any]:
        return {
            "train_examples": len(self.train_examples),
            "valid_examples": len(self.valid_examples),
            "skipped_too_long": dict(self.skipped),
            "max_seq_len": self.max_seq_len,
            "assistant_tokens_train": sum(
                sum(1 for label in labels if label != -100)
                for _, labels in self.train_examples
            ),
        }


def split_conversations(
    samples: Sequence[Dict[str, Any]], valid_ratio: float = 0.1, seed: int = 1337
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    num_valid = min(max(1, int(len(shuffled) * valid_ratio)), max(0, len(shuffled) - 1))
    return shuffled[num_valid:], shuffled[:num_valid]
