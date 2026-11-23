"""Rotary positional embeddings (RoPE), applied to Q and K only.

The cos/sin tables are cached per (device, dtype) and are not part of the
state_dict, so checkpoints stay portable and small.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cache_key: tuple | None = None

    def _build_cache(self, device: torch.device, dtype: torch.dtype, length: int) -> None:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        freqs = positions * self.inv_freq.to(device=device, dtype=torch.float32).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)  # (length, head_dim)
        # Keep the table in float32; cast on use. This avoids precision loss when
        # the model runs in bf16.
        self._cos = emb.cos().to(dtype=dtype, device=device)
        self._sin = emb.sin().to(dtype=dtype, device=device)
        self._cache_key = (str(device), dtype, length)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate q/k of shape (batch, heads, seq, head_dim) starting at ``offset``."""
        seq_len = q.shape[-2]
        needed = offset + seq_len
        if needed > self.max_seq_len:
            raise ValueError(
                f"sequence position {needed} exceeds configured max_seq_len {self.max_seq_len}"
            )
        if self._cos is None or self._cache_key != (str(q.device), q.dtype, self.max_seq_len):
            self._build_cache(q.device, q.dtype, self.max_seq_len)
        cos = self._cos[offset:needed].unsqueeze(0).unsqueeze(0)
        sin = self._sin[offset:needed].unsqueeze(0).unsqueeze(0)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
