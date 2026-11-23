"""Streaming fresh calculator conversations, so the model cannot memorise operands.

A fixed SFT file repeats the same few hundred calculator prompts every epoch; a
tiny model then memorises prompt -> expression pairs instead of learning to copy
the user's numbers into the call. This source draws new prompts (new numbers)
on every step and mixes them with the fixed conversations.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

import torch

from agent.factory import build_registry
from data.generate_sft import (
    COMPACT_SYSTEM_PROMPT,
    EVAL_PATH,
    _calc_spec,
    _norm,
    assistant_call,
    call,
    final,
    result_of,
    tool,
    user,
)

from .data import ConversationSource, conversation_to_example, pad_batch


class FreshCalcSource(ConversationSource):
    def __init__(self, *args, fresh_fraction: float = 0.4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fresh_fraction = fresh_fraction
        self._fresh_rng = random.Random(self.seed + 99)
        tmp = tempfile.mkdtemp(prefix="tinylm_fresh_")
        self._registry, _, _ = build_registry(db_path=str(Path(tmp) / "f.db"))
        self._held = [
            _norm(json.loads(l)["prompt"]) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
        ]

    def _fresh_example(self) -> Tuple[List[int], List[int]]:
        rng = self._fresh_rng
        while True:
            if rng.random() < 0.06:
                n = rng.randint(2, 900)
                prompt = rng.choice(
                    [f"What is {n} divided by zero?", f"Compute {n} / 0.", f"Divide {n} by 0 for me.", f"How much is {n} over 0?"]
                )
                expr, template = f"{n} / 0", None
            else:
                prompt, expr, template = _calc_spec(rng)
            norm = _norm(prompt)
            if any(h in norm or norm in h for h in self._held):
                continue
            out = self._registry.execute("calculator", {"expression": expr}).to_json()
            if template is None:
                answer = "That is undefined: division by zero has no value."
            else:
                res = result_of(out)
                if res is None:
                    continue
                answer = template.format(r=res)
            messages = [
                {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
                user(prompt),
                assistant_call(call("calculator", expression=expr)),
                tool(out),
                final(answer),
            ]
            ids, labels = conversation_to_example(self.tokenizer, messages)
            if len(ids) <= self.max_seq_len:
                return ids, labels

    def train_batches(self, batch_size: int, max_steps: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        fixed = super().train_batches(1, max_steps)
        while True:
            examples = []
            for _ in range(batch_size):
                if self._fresh_rng.random() < self.fresh_fraction:
                    examples.append(self._fresh_example())
                else:
                    ids, labels = next(fixed)
                    row_ids = ids[0][ids[0] != self.pad_token_id].tolist()
                    row_labels = labels[0][: len(row_ids)].tolist()
                    examples.append((row_ids, row_labels))
            yield pad_batch(examples, self.pad_token_id)
