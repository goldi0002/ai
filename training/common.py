"""Shared utilities for the TinyLM-50 training loops.

Everything here is deliberately small and explicit: device/precision choice,
learning-rate schedule, checkpoint format and JSONL logging. Both the
pretraining loop (``training.train``) and the instruction-tuning loop
(``training.sft``) use it, so a checkpoint from either can be resumed by the
same code.
"""

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml

CHECKPOINT_FORMAT_VERSION = 1


# --------------------------------------------------------------------- config
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


# ------------------------------------------------------------------- runtime
def pick_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str = "auto", device: torch.device | None = None) -> torch.dtype:
    """Map a precision name to a torch dtype.

    ``auto`` means bf16 on CUDA when supported, otherwise float32. CPU
    training in float32 is the safe default: it is numerically stable and, on
    the small CPU runs this project is developed on, autocast brings no
    measurable speedup.
    """
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name == "auto":
        if device is not None and device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device is not None and device.type == "cuda":
            return torch.float16
        return torch.float32
    raise ValueError(f"unknown precision {name!r}; expected auto|bf16|fp16|fp32")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ schedule
def cosine_lr(
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_ratio: float = 0.02,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup followed by cosine decay to ``base_lr * min_lr_ratio``."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    warmup_steps = max(1, int(round(warmup_ratio * max_steps)))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    scale = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * scale


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> torch.optim.AdamW:
    """AdamW with no weight decay on norms, biases or scalars (GPT convention)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(beta1, beta2))


# ------------------------------------------------------------------ checkpoint
def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    model_config: Any,
    train_config: Dict[str, Any],
    tokenizer_meta: Optional[Dict[str, Any]] = None,
    tokens_seen: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a self-describing checkpoint.

    The model config is stored inline so a checkpoint can be rebuilt without
    the YAML file that produced it. Tokenizer ids are stored too, and loading
    rejects a checkpoint whose ids do not match the tokenizer.
    """
    path = Path(path)
    unwrapped = getattr(model, "_orig_mod", model)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "model_config": model_config.to_dict() if hasattr(model_config, "to_dict") else dict(model_config),
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "train_config": dict(train_config),
        "tokenizer_meta": tokenizer_meta or {},
        "parameter_count": count_parameters(unwrapped),
        "extra": extra or {},
    }
    _atomic_torch_save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "model_config" not in payload:
        raise ValueError(f"{path} is not a TinyLM checkpoint")
    if payload.get("format_version", 0) > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {payload['format_version']} is newer than supported "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )
    return payload


def checkpoint_matches_tokenizer(payload: Dict[str, Any], config_ids: Dict[str, int]) -> bool:
    stored = (payload.get("tokenizer_meta") or {}).get("special_ids") or {}
    return all(stored.get(token) == value for token, value in config_ids.items()) if stored else True


# -------------------------------------------------------------------- logging
def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class AverageMeter:
    """Weighted running average (used for loss and gradient-norm reporting)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * weight
        self.count += weight

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(remainder):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"
