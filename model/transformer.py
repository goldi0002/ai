"""A single pre-norm decoder block."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .config import TinyLMConfig
from .feedforward import SwiGLU
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm residual block: attn(norm(x)) then mlp(norm(x))."""

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.ffn_hidden_size, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attn_out)
        hidden_states = hidden_states + self.dropout(self.ffn(self.ffn_norm(hidden_states)))
        return hidden_states, present
