"""Synthetic copy curriculum: teach the model to copy spans from its context.

    python -m training.copy_task --init checkpoints/pretrain_base.pt \
        --max-steps 600 --out-dir checkpoints/copy

Why: after a tiny pretrain, SFT teaches a model *which* tool to call and the JSON
shape, but not to copy the user's numbers into the arguments (measured: 0% of
fresh calculator prompts reproduced the prompt's numbers). Copying is an
induction-head behaviour that normally emerges from far more data than this
project has. Repeated-span data is the cheapest known way to form it, so this
stage runs between pretraining and SFT.

Each example shows a span once and then repeats it. The loss is applied only to
the repeat, so every gradient step pushes the model to look back in context:

* ``random``  — random ordinary token ids repeated verbatim (pure induction);
* ``numbers`` — a list of random integers/decimals written with one separator
  and then re-written with another, using the real tokenizer so digit
  tokenisation matches what the calculator prompts need;
* ``phrase``  — a short run of words repeated after a different lead-in.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import torch

from model.config import TinyLMConfig
from model.model import TinyLM
from model.tokenizer import TinyTokenizer

from .common import load_checkpoint, load_yaml, pick_device, pick_dtype, section, set_seed
from .data import pad_batch
from .trainer import Trainer, TrainingConfig

WORDS = (
    "alpha beta gamma delta river stone cloud paper window garden orange silver forest "
    "engine bridge candle market planet ticket violet anchor basket copper harbor lantern "
    "meadow pillow rocket saddle thunder velvet wagon yellow zipper"
).split()
FIRST_ORDINARY_ID = 12  # ids below are pad/unk/bos/eos and the chat/tool specials


class CopySource:
    """Infinite, deterministic stream of copy examples with a fixed eval set."""

    def __init__(self, tokenizer: TinyTokenizer, max_len: int = 96, seed: int = 1337) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.pad = tokenizer.special_ids["<pad>"]
        self.bos = tokenizer.id("<bos>")
        self.eos = tokenizer.id("<eos>")
        self.sep = tokenizer.id("<evidence>")
        self.vocab = tokenizer.vocab_size
        self._rng = random.Random(seed)
        self.epochs = 0

    # -- example builders ----------------------------------------------------
    def _random(self, rng: random.Random) -> Tuple[List[int], List[int]]:
        span = [rng.randrange(FIRST_ORDINARY_ID, self.vocab) for _ in range(rng.randint(4, 24))]
        return span, span

    def _numbers(self, rng: random.Random) -> Tuple[List[int], List[int]]:
        count = rng.randint(2, 5)
        nums = []
        for _ in range(count):
            roll = rng.random()
            if roll < 0.5:
                nums.append(str(rng.randint(2, 99)))
            elif roll < 0.85:
                nums.append(str(rng.randint(100, 99999)))
            else:
                nums.append(f"{rng.randint(0, 99)}.{rng.randint(1, 9)}")
        first = rng.choice([", ", " ", " and ", "; "]).join(nums)
        second = rng.choice([" ", " + ", " * ", " / ", ", ", " - "]).join(nums)
        lead = rng.choice(["", "numbers: ", "Compute ", "values "])
        tail = rng.choice(["", "expression: ", "=> ", "answer: "])
        return self.tokenizer.encode(lead + first), self.tokenizer.encode(tail + second)

    def _phrase(self, rng: random.Random) -> Tuple[List[int], List[int]]:
        words = [rng.choice(WORDS) for _ in range(rng.randint(2, 6))]
        text = " ".join(words)
        return self.tokenizer.encode(text), self.tokenizer.encode(rng.choice(["", "again: ", "repeat: "]) + text)

    def example(self, rng: random.Random) -> Tuple[List[int], List[int]]:
        kind = rng.choices(["random", "numbers", "phrase"], weights=[0.4, 0.45, 0.15])[0]
        shown, repeated = getattr(self, "_" + kind)(rng)
        ids = [self.bos] + list(shown) + [self.sep] + list(repeated) + [self.eos]
        start = 1 + len(shown) + 1  # first position of the repeat
        mask = [False] * start + [True] * (len(repeated) + 1)
        ids, mask = ids[: self.max_len], mask[: self.max_len]
        inputs = ids[:-1]
        labels = [ids[i + 1] if mask[i + 1] else -100 for i in range(len(inputs))]
        return inputs, labels

    # -- trainer interface ---------------------------------------------------
    def train_batches(self, batch_size: int, max_steps: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield pad_batch([self.example(self._rng) for _ in range(batch_size)], self.pad)

    def eval_batches(self, batch_size: int, num_batches: int, split: str = "valid"):
        rng = random.Random(self.seed + 4242)
        return [
            pad_batch([self.example(rng) for _ in range(batch_size)], self.pad)
            for _ in range(num_batches)
        ]

    def metadata(self) -> Dict[str, Any]:
        return {"kind": "synthetic copy curriculum", "max_len": self.max_len, "streams": ["random", "numbers", "phrase"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyLM-50 synthetic copy curriculum")
    parser.add_argument("--config", default="configs/tiny50.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/tiny50_sp.model")
    parser.add_argument("--init", required=True, help="checkpoint to start from")
    parser.add_argument("--out-dir", default="checkpoints/copy")
    parser.add_argument("--run-name", default="copy")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = load_yaml(args.config)
    model_cfg = TinyLMConfig.from_dict(dict(section(raw, "model")))
    tokenizer = TinyTokenizer(args.tokenizer)
    tokenizer.apply_to_config(model_cfg)
    payload = load_checkpoint(args.init)
    model_cfg = TinyLMConfig.from_dict(payload["model_config"])
    if model_cfg.vocab_size != tokenizer.vocab_size:
        print("error: checkpoint vocab does not match the tokenizer", file=sys.stderr)
        return 1

    train_cfg = TrainingConfig.from_dict(
        {
            "batch_size": args.batch_size,
            "grad_accum_steps": 1,
            "max_steps": args.max_steps,
            "learning_rate": args.lr,
            "warmup_ratio": 0.05,
            "weight_decay": 0.0,
            "log_every": args.log_every,
            "eval_every": args.eval_every,
            "eval_batches": 4,
            "checkpoint_every": max(args.max_steps, 1),
            "checkpoint_dir": args.out_dir,
            "max_seq_len": args.max_len,
            "seed": args.seed,
        },
        run_name=args.run_name,
    )
    source = CopySource(tokenizer, max_len=args.max_len, seed=args.seed)
    model = TinyLM(model_cfg)
    model.load_state_dict(payload["model"])
    set_seed(args.seed)
    device = pick_device(args.device)
    trainer = Trainer(model, source, train_cfg, device=device, dtype=pick_dtype(train_cfg.precision, device))
    print(json.dumps({"device": str(device), "init": args.init, "source": source.metadata()}, indent=2), flush=True)
    print(json.dumps(trainer.train(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
