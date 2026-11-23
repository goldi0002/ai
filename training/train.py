"""Pretraining CLI for TinyLM-50.

    python -m training.train --config configs/tiny50.yaml --max-steps 100
    python -m training.train --config configs/tiny50.yaml --resume checkpoints/latest.pt

The loop is real: it optimises next-token cross-entropy on packed data and
writes a JSONL log. Nothing here downloads data or fakes metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch

from data.pack import PackedDataset
from model.config import TinyLMConfig
from model.model import TinyLM
from model.tokenizer import TinyTokenizer

from .common import load_checkpoint, load_yaml, pick_device, pick_dtype, section, set_seed
from .data import PretrainSource
from .trainer import Trainer, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyLM-50 pretraining")
    parser.add_argument("--config", default="configs/tiny50.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/tiny50_sp.model")
    parser.add_argument("--processed-dir", default="datasets/processed")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--run-name", default="pretrain")
    parser.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="override micro batch size")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None, help="must match the packed data")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-eval", action="store_true")
    return parser


def _resolve_tokenizer(path: str | Path, model_cfg: TinyLMConfig) -> TinyTokenizer:
    tokenizer = TinyTokenizer(path)
    tokenizer.verify(model_cfg.vocab_size)
    tokenizer.apply_to_config(model_cfg)
    return tokenizer


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = load_yaml(args.config)
    model_cfg = TinyLMConfig.from_dict(dict(section(raw, "model")))
    train_data = dict(section(raw, "training"))
    if args.max_steps is not None:
        train_data["max_steps"] = args.max_steps
    if args.batch_size is not None:
        train_data["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        train_data["grad_accum_steps"] = args.grad_accum
    if args.lr is not None:
        train_data["learning_rate"] = args.lr
    if args.eval_every is not None:
        train_data["eval_every"] = args.eval_every
    if args.checkpoint_every is not None:
        train_data["checkpoint_every"] = args.checkpoint_every
    if args.log_every is not None:
        train_data["log_every"] = args.log_every
    if args.precision is not None:
        train_data["precision"] = args.precision
    if args.no_eval:
        train_data["eval_every"] = 0
    train_cfg = TrainingConfig.from_dict(train_data, run_name=args.run_name)
    train_cfg.checkpoint_dir = args.out_dir

    processed = Path(args.processed_dir)
    train_path = processed / "train_tokens.bin"
    valid_path = processed / "valid_tokens.bin"
    if not train_path.exists():
        print(f"error: missing {train_path}; run `python -m data.prepare pack`", file=sys.stderr)
        return 1
    train_packed = PackedDataset.load(train_path)
    valid_packed = PackedDataset.load(valid_path) if valid_path.exists() else None
    if args.seq_len is not None and args.seq_len != train_packed.seq_len:
        print(
            f"error: --seq-len {args.seq_len} != packed seq_len {train_packed.seq_len}",
            file=sys.stderr,
        )
        return 1
    model_cfg.max_seq_len = train_packed.seq_len

    tokenizer = _resolve_tokenizer(args.tokenizer, model_cfg)
    set_seed(train_cfg.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(train_cfg.precision, device)

    resume_payload = None
    if args.resume:
        resume_payload = load_checkpoint(args.resume)
        saved_cfg = TinyLMConfig.from_dict(resume_payload["model_config"])
        if saved_cfg.vocab_size != model_cfg.vocab_size:
            print(
                f"error: checkpoint vocab {saved_cfg.vocab_size} != tokenizer vocab "
                f"{model_cfg.vocab_size}",
                file=sys.stderr,
            )
            return 1
        model_cfg = saved_cfg

    model = TinyLM(model_cfg)
    source = PretrainSource(train_packed, valid_packed, seed=train_cfg.seed)
    source.tokenizer_meta = {"special_ids": tokenizer.special_ids, "vocab_size": tokenizer.vocab_size}
    trainer = Trainer(model, source, train_cfg, device=device, dtype=dtype)

    resume_step, tokens_seen = 0, 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"])
        trainer.load_optimizer_state(resume_payload.get("optimizer"))
        resume_step = int(resume_payload.get("step", 0))
        tokens_seen = int(resume_payload.get("tokens_seen", 0))
        print(f"resumed from {args.resume} at step {resume_step} ({tokens_seen:,} tokens)")

    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(dtype),
                "parameters": model.num_parameters(),
                "breakdown": model.parameter_breakdown(),
                "train_sequences": len(train_packed),
                "valid_sequences": len(valid_packed) if valid_packed else 0,
                "seq_len": train_packed.seq_len,
                "effective_batch_tokens": train_cfg.batch_size
                * train_cfg.grad_accum_steps
                * (train_packed.seq_len - 1),
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
