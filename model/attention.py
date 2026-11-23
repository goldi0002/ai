"""Causal grouped-query attention using PyTorch scaled dot-product attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyLMConfig
from .rotary import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Multi-head attention where K/V have fewer heads than Q.

    ``num_heads`` query heads share ``num_kv_heads`` key/value heads; each KV
    head is repeated ``num_heads // num_kv_heads`` times, which keeps KV cache
    small while preserving full query capacity.

    Returns (output, present_key_value). ``present_key_value`` is None unless
    ``use_cache`` is True.
    """

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = hidden_states.shape
        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)  # (b, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rotary(q, k, offset=past_len)

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)
        present = (k, v) if use_cache else None

        k_expanded = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_key_value_groups, dim=1)

        kv_len = k.shape[2]
        dropout_p = self.dropout if self.training else 0.0
        # With a KV cache we must build the mask ourselves: is_causal=True
        # aligns at the top-left, so a single new query would attend only to
        # the first cached key instead of the whole prefix.
        needs_explicit_mask = attention_mask is not None or past_len > 0

        if needs_explicit_mask:
            # Build a boolean (b, 1, seq, kv) mask: True = may attend.
            if attention_mask is not None:
                key_valid = attention_mask[:, None, None, :].to(torch.bool)
            else:
                key_valid = torch.ones(
                    (batch, 1, 1, kv_len), dtype=torch.bool, device=hidden_states.device
                )
            q_pos = torch.arange(past_len, past_len + seq_len, device=hidden_states.device)
            k_pos = torch.arange(kv_len, device=hidden_states.device)
            causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # (seq, kv)
            bool_mask = key_valid & causal.unsqueeze(0).unsqueeze(0)
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded, attn_mask=bool_mask, dropout_p=dropout_p
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded, is_causal=True, dropout_p=dropout_p
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output), present
