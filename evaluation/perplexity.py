"""Perplexity / cross-entropy evaluation on packed token files."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from data.pack import PackedDataset


@torch.no_grad()
def evaluate_packed(
    model: torch.nn.Module,
    packed: PackedDataset,
    batch_size: int = 2,
    max_batches: int = 10,
    device: str | torch.device = "cpu",
    seed: int = 1234,
    seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    if len(packed) == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "tokens": 0, "batches": 0}
    device = torch.device(device)
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for _ in range(max_batches):
        indices = torch.randint(0, len(packed), (min(batch_size, len(packed)),), generator=generator)
        array = packed.batch(indices.tolist())
        ids = torch.from_numpy(array.astype("int64")).to(device)
        if seq_len is not None:
            ids = ids[:, : seq_len + 1]
        input_ids, labels = ids[:, :-1], ids[:, 1:]
        out = model(input_ids, targets=labels)
        tokens = labels.numel()
        total_loss += float(out.loss) * tokens
        total_tokens += tokens
        batches += 1
    loss = total_loss / max(1, total_tokens)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "tokens": total_tokens,
        "batches": batches,
        "seq_len": int(input_ids.shape[1]),
    }
