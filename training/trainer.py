"""The training loop shared by pretraining and instruction tuning."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

from model.config import TinyLMConfig
from model.model import TinyLM

from .common import (
    AverageMeter,
    append_jsonl,
    build_optimizer,
    cosine_lr,
    count_parameters,
    format_seconds,
    pick_dtype,
    save_checkpoint,
)


@dataclass
class TrainingConfig:
    batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 20000
    warmup_ratio: float = 0.02
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "auto"
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 20
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    seed: int = 1337
    gradient_checkpointing: bool = False
    compile: bool = False
    num_workers: int = 2
    max_seq_len: int = 1024
    run_name: str = "train"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], run_name: str = "train") -> "TrainingConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        return cls(run_name=run_name, **data)

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}  # type: ignore[attr-defined]


@dataclass
class EvalResult:
    loss: float
    perplexity: float
    batches: int

    def as_dict(self) -> Dict[str, Any]:
        return {"loss": self.loss, "perplexity": self.perplexity, "batches": self.batches}


class Trainer:
    """Micro-batch + gradient-accumulation loop with cosine schedule."""

    def __init__(
        self,
        model: TinyLM,
        source: Any,
        config: TrainingConfig,
        device: torch.device,
        dtype: torch.dtype | None = None,
        log_dir: str | Path = ".logs",
    ) -> None:
        self.model = model
        self.source = source
        self.config = config
        self.device = device
        self.dtype = dtype or pick_dtype(config.precision, device)
        self.log_dir = Path(log_dir)
        self.log_path = Path(config.checkpoint_dir) / f"{config.run_name}_log.jsonl"
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.step = 0
        self.tokens_seen = 0
        self.best_val: Optional[float] = None
        self.history: list[Dict[str, Any]] = []

        self.model.to(device)
        self.model.gradient_checkpointing = config.gradient_checkpointing
        self.optimizer = build_optimizer(
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.dtype == torch.float16 and device.type == "cuda"
        )
        self.amp_enabled = self.dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"
        self.amp_dtype = self.dtype if self.amp_enabled else torch.float32

    # ------------------------------------------------------------------ utils
    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    def _step_lr(self, step: int) -> float:
        return cosine_lr(
            step,
            self.config.max_steps,
            self.config.learning_rate,
            self.config.warmup_ratio,
            self.config.min_lr_ratio,
        )

    def evaluate(self, split: str = "valid") -> EvalResult:
        batches = self.source.eval_batches(
            self.config.batch_size, self.config.eval_batches, split=split
        )
        if not batches:
            return EvalResult(loss=float("nan"), perplexity=float("nan"), batches=0)
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for input_ids, labels in batches:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                with self._autocast():
                    out = self.model(input_ids, targets=labels)
                tokens = int((labels != -100).sum())
                total_loss += float(out.loss) * tokens
                total_tokens += tokens
        if was_training:
            self.model.train()
        mean_loss = total_loss / max(1, total_tokens)
        return EvalResult(loss=mean_loss, perplexity=math.exp(min(mean_loss, 20.0)), batches=len(batches))

    # --------------------------------------------------------------- training
    def train(self, resume_step: int = 0, tokens_seen: int = 0, log_every: int | None = None) -> Dict[str, Any]:
        cfg = self.config
        self.step = resume_step
        self.tokens_seen = tokens_seen
        log_every = log_every or cfg.log_every
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        batches: Iterator[Tuple[torch.Tensor, torch.Tensor]] = self.source.train_batches(
            cfg.batch_size, cfg.max_steps
        )
        self.model.train()
        started = time.time()
        last_log_time = started
        last_log_tokens = self.tokens_seen
        loss_meter = AverageMeter()
        grad_meter = AverageMeter()
        log_path = self.log_path
        step_losses = []

        for step in range(resume_step, cfg.max_steps):
            lr = self._step_lr(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            micro_tokens = 0
            for _ in range(cfg.grad_accum_steps):
                input_ids, labels = next(batches)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                with self._autocast():
                    out = self.model(input_ids, targets=labels)
                    loss = out.loss / cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                micro_losses.append(out.loss.detach().item())
                micro_tokens += int((labels != -100).sum())
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step = step + 1
            self.tokens_seen += micro_tokens
            step_loss = sum(micro_losses) / len(micro_losses)
            loss_meter.update(step_loss, micro_tokens)
            grad_meter.update(float(grad_norm), 1)
            step_losses.append(step_loss)

            is_last = self.step >= cfg.max_steps
            if self.step % log_every == 0 or is_last:
                now = time.time()
                interval_tokens = self.tokens_seen - last_log_tokens
                interval_seconds = max(1e-9, now - last_log_time)
                elapsed = now - started
                remaining = (cfg.max_steps - self.step) * (elapsed / max(1, self.step - resume_step))
                record = {
                    "run": cfg.run_name,
                    "step": self.step,
                    "max_steps": cfg.max_steps,
                    "lr": lr,
                    "train_loss": loss_meter.average,
                    "grad_norm": grad_meter.average,
                    "tokens_seen": self.tokens_seen,
                    "tokens_per_sec": interval_tokens / interval_seconds,
                    "elapsed_sec": elapsed,
                    "eta": format_seconds(remaining),
                }
                append_jsonl(log_path, record)
                self.history.append(record)
                print(
                    f"[{cfg.run_name}] step {self.step}/{cfg.max_steps} "
                    f"loss {record['train_loss']:.4f} lr {lr:.2e} "
                    f"gnorm {record['grad_norm']:.2f} "
                    f"{record['tokens_per_sec']:.1f} tok/s eta {record['eta']}",
                    flush=True,
                )
                last_log_time, last_log_tokens = now, self.tokens_seen
                loss_meter.reset()
                grad_meter.reset()

            if cfg.eval_every and self.step % cfg.eval_every == 0:
                result = self.evaluate()
                if result.batches:
                    best = self.best_val is None or result.loss < self.best_val
                    if best:
                        self.best_val = result.loss
                    append_jsonl(log_path, {"run": cfg.run_name, "step": self.step, "eval": result.as_dict()})
                    print(
                        f"[{cfg.run_name}] eval step {self.step}: "
                        f"loss {result.loss:.4f} ppl {result.perplexity:.2f}",
                        flush=True,
                    )
                    self.save(f"step_{self.step}.pt", extra={"eval_loss": result.loss})
                    if best:
                        self.save("best.pt", extra={"eval_loss": result.loss})

            if cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0 and not is_last:
                self.save(f"step_{self.step}.pt")

        final_eval = self.evaluate()
        self.save("latest.pt", extra={"eval_loss": final_eval.loss if final_eval.batches else None})
        summary = {
            "run": self.config.run_name,
            "steps": self.step,
            "tokens_seen": self.tokens_seen,
            "parameters": count_parameters(self.model),
            "elapsed_sec": time.time() - started,
            "final_train_loss": step_losses[-1] if step_losses else None,
            "first_train_loss": step_losses[0] if step_losses else None,
            "eval": final_eval.as_dict(),
            "checkpoint": str(self.checkpoint_dir / "latest.pt"),
        }
        append_jsonl(log_path, {"run": cfg.run_name, "summary": summary})
        return summary

    # ------------------------------------------------------------- checkpoint
    def save(self, name: str, extra: Optional[Dict[str, Any]] = None) -> Path:
        path = self.checkpoint_dir / name
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            model_config=self.model.config,
            train_config=self.config.to_dict(),
            tokenizer_meta=getattr(self.source, "tokenizer_meta", None)
            or (getattr(self.source, "metadata", lambda: {})()),
            tokens_seen=self.tokens_seen,
            extra=extra,
        )

    def load_optimizer_state(self, state: Dict[str, Any]) -> None:
        if state:
            self.optimizer.load_state_dict(state)
