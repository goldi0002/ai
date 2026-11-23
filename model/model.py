"""TinyLM-50: a ~49M parameter decoder-only transformer with GQA and SwiGLU.

Public API
----------
>>> cfg = TinyLMConfig()
>>> model = TinyLM(cfg)
>>> out = model(input_ids)                      # ModelOutput(logits=..., ...)
>>> out = model(input_ids, targets=labels)      # .loss is the mean NLL
>>> ids = model.generate(input_ids, max_new_tokens=64, temperature=0.7, top_p=0.9)

``forward`` returns a :class:`ModelOutput` rather than a bare tensor so that
loss, logits and the KV cache can travel together without an ambiguous tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyLMConfig
from .embeddings import TokenEmbedding
from .normalization import RMSNorm
from .transformer import TransformerBlock

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[KVCache] = None


class TinyLM(nn.Module):
    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(config)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding.weight

        self.gradient_checkpointing = config.gradient_checkpointing
        self.apply(self._init_weights)
        # Residual output projections are scaled by 1/sqrt(2 * num_layers) so the
        # residual stream variance stays bounded in a 14-layer stack.
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=config.initializer_range / (2 * config.num_layers) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    # ---------------------------------------------------------------- stats
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def parameter_breakdown(self) -> dict:
        embedding = self.embed_tokens.embedding.weight.numel()
        blocks = sum(p.numel() for p in self.layers.parameters())
        final_norm = sum(p.numel() for p in self.norm.parameters())
        head = 0 if self.config.tie_embeddings else self.lm_head.weight.numel()
        total = self.num_parameters()
        return {
            "embedding": embedding,
            "blocks": blocks,
            "final_norm": final_norm,
            "lm_head": head,
            "total": total,
            "non_embedding": total - embedding,
        }

    # -------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[KVCache] = None,
    ) -> ModelOutput:
        """Run the model.

        Args:
            input_ids: (batch, seq) int64 token ids.
            targets: (batch, seq) int64 next-token labels; -100 means "ignore".
            attention_mask: (batch, kv_len) bool/int, 1 = valid token.
            use_cache: return per-layer (k, v) for incremental decoding.
            past_key_values: cache from a previous forward.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (batch, seq), got {tuple(input_ids.shape)}")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
            if past_key_values is not None:
                past_len = past_key_values[0][0].shape[-2]
                attention_mask = F.pad(attention_mask, (past_len, 0), value=True)

        hidden_states = self.embed_tokens(input_ids)
        presents: KVCache | None = [] if use_cache else None

        for index, layer in enumerate(self.layers):
            past = past_key_values[index] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                # Checkpointing needs a custom autograd function to thread the KV
                # cache, which is only used at inference; cache is ignored here.
                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    None,
                    False,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past,
                    use_cache=use_cache,
                )
            if presents is not None and present is not None:
                presents.append(present)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return ModelOutput(logits=logits, loss=loss, past_key_values=presents)

    # ------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        stop_token_ids: Optional[set[int]] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling; returns prompt + generated token ids.

        temperature <= 0 means greedy decoding. top_p is nucleus sampling on the
        filtered distribution; top_k is applied before top_p when > 0.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be (batch, seq)")
        if max_new_tokens <= 0:
            return input_ids
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        stop = set(stop_token_ids or ())
        stop.add(eos)

        ids = input_ids
        past: KVCache | None = None
        finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
        generated = []

        for _ in range(max_new_tokens):
            if past is not None and use_cache:
                step_ids = ids[:, -1:]
            else:
                step_ids = ids[:, -(self.config.max_seq_len - 1):] if ids.size(1) > self.config.max_seq_len else ids
            out = self.forward(step_ids, use_cache=use_cache, past_key_values=past)
            logits = out.logits[:, -1, :].float()
            past = out.past_key_values

            if repetition_penalty and repetition_penalty != 1.0 and ids.size(1) > 0:
                for batch_index in range(ids.size(0)):
                    seen = torch.unique(ids[batch_index])
                    scores = logits[batch_index, seen]
                    logits[batch_index, seen] = torch.where(
                        scores > 0, scores / repetition_penalty, scores * repetition_penalty
                    )

            if temperature is None or temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                    logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
                if top_p and 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative = probs.cumsum(dim=-1)
                    remove = cumulative - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    probs = F.softmax(sorted_logits, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, sampled)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_token], dim=-1)
            generated.append(next_token)
            finished |= torch.isin(next_token.squeeze(-1), torch.tensor(sorted(stop), device=ids.device))
            if bool(finished.all()):
                break

        return ids
