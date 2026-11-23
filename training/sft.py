"""Supervised instruction tuning (SFT) on conversation JSONL files.

    python -m training.sft --config configs/tiny50.yaml \
        --data datasets/instructions/tiny50_sft.jsonl \
        --init checkpoints/latest.pt --max-steps 200

Loss is masked to assistant tokens (including tool-call JSON) only; system,
user and tool-result tokens provide context but are never targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from model.config import TinyLMConfig
from model.model import TinyLM
from model.tokenizer import TinyTokenizer

from .common import load_checkpoint, load_yaml, pick_device, pick_dtype, section, set_seed
from .data import ConversationSource, read_conversations, split_conversations
from .fresh_calc import FreshCalcSource
from .trainer import Trainer, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyLM-50 instruction tuning")
    parser.add_argument("--config", default="configs/tiny50.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/tiny50_sp.model")
    parser.add_argument("--data", default="datasets/instructions/tiny50_sft.jsonl")
    parser.add_argument("--init", default=None, help="checkpoint to start from")
    parser.add_argument("--resume", default=None, help="checkpoint with optimiser state")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--run-name", default="sft")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--fresh-calc", type=float, default=0.0, help="fraction of each batch drawn from a stream of freshly generated calculator conversations")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = load_yaml(args.config)
    model_cfg = TinyLMConfig.from_dict(dict(section(raw, "model")))
    sft_data = dict(section(raw, "sft"))
    for key, value in (
        ("max_steps", args.max_steps),
        ("batch_size", args.batch_size),
        ("grad_accum_steps", args.grad_accum),
        ("learning_rate", args.lr),
        ("eval_every", args.eval_every),
        ("log_every", args.log_every),
        ("checkpoint_every", args.checkpoint_every),
        ("precision", args.precision),
    ):
        if value is not None:
            sft_data[key] = value
    sft_data.setdefault("seed", args.seed)
    sft_data.setdefault("max_seq_len", model_cfg.max_seq_len)
    train_cfg = TrainingConfig.from_dict(sft_data, run_name=args.run_name)
    train_cfg.checkpoint_dir = args.out_dir

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: missing instruction data {data_path}", file=sys.stderr)
        return 1
    tokenizer = TinyTokenizer(args.tokenizer)
    tokenizer.apply_to_config(model_cfg)
    samples = read_conversations(data_path)
    train_samples, valid_samples = split_conversations(
        samples, valid_ratio=args.valid_ratio, seed=args.seed
    )

    start_payload = None
    if args.resume:
        start_payload = load_checkpoint(args.resume)
    elif args.init:
        start_payload = load_checkpoint(args.init)
    if start_payload is not None:
        saved_cfg = TinyLMConfig.from_dict(start_payload["model_config"])
        if saved_cfg.vocab_size != model_cfg.vocab_size:
            print(
                f"error: checkpoint vocab {saved_cfg.vocab_size} != tokenizer vocab "
                f"{model_cfg.vocab_size}",
                file=sys.stderr,
            )
            return 1
        model_cfg = saved_cfg

    source_kwargs = dict(max_seq_len=min(train_cfg.max_seq_len, model_cfg.max_seq_len), seed=args.seed)
    if args.fresh_calc > 0:
        source = FreshCalcSource(train_samples, valid_samples, tokenizer, fresh_fraction=args.fresh_calc, **source_kwargs)
    else:
        source = ConversationSource(train_samples, valid_samples, tokenizer, **source_kwargs)
    source.tokenizer_meta = {"special_ids": tokenizer.special_ids, "vocab_size": tokenizer.vocab_size}
    model = TinyLM(model_cfg)
    if start_payload is not None:
        model.load_state_dict(start_payload["model"])

    set_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(train_cfg.precision, device)
    trainer = Trainer(model, source, train_cfg, device=device, dtype=dtype)

    resume_step, tokens_seen = 0, 0
    if args.resume and start_payload is not None:
        trainer.load_optimizer_state(start_payload.get("optimizer"))
        resume_step = int(start_payload.get("step", 0))
        tokens_seen = int(start_payload.get("tokens_seen", 0))

    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(dtype),
                "parameters": model.num_parameters(),
                "data": str(data_path),
                "sources": source.metadata(),
                "init": args.init or args.resume,
            },
            indent=2,
        ),
        flush=True,
    )
    summary = trainer.train(resume_step=resume_step, tokens_seen=tokens_seen)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
