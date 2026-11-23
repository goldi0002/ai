"""Load a TinyLM-50 checkpoint and generate text.

The loader refuses mismatched tokenizers: a checkpoint whose special-token ids
or vocabulary differ from the tokenizer is a corruption, not a warning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from model.config import TinyLMConfig
from model.model import TinyLM
from model.tokenizer import TinyTokenizer
from training.common import load_checkpoint, pick_device, pick_dtype


@dataclass
class LoadedModel:
    model: TinyLM
    tokenizer: TinyTokenizer
    config: TinyLMConfig
    device: torch.device
    checkpoint_path: Path
    step: int
    tokenizer_meta: Dict[str, Any]

    def info(self) -> Dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "step": self.step,
            "parameters": self.model.num_parameters(),
            "breakdown": self.model.parameter_breakdown(),
            "vocab_size": self.config.vocab_size,
            "max_seq_len": self.config.max_seq_len,
            "device": str(self.device),
            "tokenizer": str(self.tokenizer.model_path),
        }


def load_model(
    checkpoint_path: str | Path = "checkpoints/latest.pt",
    tokenizer_path: str | Path = "data/tokenizer/tiny50_sp.model",
    device: str = "auto",
    precision: str = "auto",
) -> LoadedModel:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Train one with "
            "`python -m training.train --max-steps 50`."
        )
    payload = load_checkpoint(checkpoint_path)
    config = TinyLMConfig.from_dict(payload["model_config"])
    tokenizer = TinyTokenizer(tokenizer_path)
    tokenizer.verify(config.vocab_size)
    stored_ids = (payload.get("tokenizer_meta") or {}).get("special_ids") or {}
    for token, expected in tokenizer.special_ids.items():
        if token in stored_ids and stored_ids[token] != expected:
            raise ValueError(
                f"tokenizer id mismatch for {token}: checkpoint {stored_ids[token]} "
                f"!= tokenizer {expected}"
            )
    resolved_device = pick_device(device)
    dtype = pick_dtype(precision, resolved_device)
    model = TinyLM(config)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if unexpected:
        raise ValueError(f"unexpected checkpoint tensors: {unexpected}")
    model.to(device=resolved_device)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        step=int(payload.get("step", 0)),
        tokenizer_meta=payload.get("tokenizer_meta") or {},
    )


@torch.no_grad()
def generate_text(
    loaded: LoadedModel,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    add_bos: bool = True,
) -> Dict[str, Any]:
    prompt_ids = loaded.tokenizer.encode(prompt, add_bos=add_bos)
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    limit = loaded.config.max_seq_len - 1
    if len(prompt_ids) > limit:
        prompt_ids = prompt_ids[-limit:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.device)
    started = time.perf_counter()
    output = loaded.model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    elapsed = time.perf_counter() - started
    new_ids = output[0, len(prompt_ids) :].tolist()
    return {
        "text": loaded.tokenizer.decode(new_ids, skip_special_tokens=True),
        "raw_text": loaded.tokenizer.decode(new_ids, skip_special_tokens=False),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(new_ids),
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(new_ids) / elapsed if elapsed > 0 else float("inf"),
        "prompt": prompt,
    }
