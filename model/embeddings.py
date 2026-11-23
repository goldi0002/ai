"""Token embedding table (shared with the LM head)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TinyLMConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding.padding_idx = config.pad_token_id
        self.scale = config.hidden_size**0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Scaling the embedding by sqrt(hidden) keeps activation variance stable
        # at initialisation for pre-norm residual stacks.
        return self.embedding(input_ids) * self.scale
