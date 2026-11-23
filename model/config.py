"""Configuration object for the TinyLM-50 decoder-only transformer.

Model configuration is deliberately independent from training and runtime
configuration: a checkpoint stores exactly this object, so a saved model can be
rebuilt without knowing how it was trained.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Order matters: SentencePiece assigns ids to user-defined symbols in order, and
# the tokenizer verifies these ids before the model is built. ``<unk>`` is a
# SentencePiece meta token, so its id is the last vocabulary id.
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<user>",
    "<assistant>",
    "<system>",
    "<tool_call>",
    "<tool_result>",
    "<research>",
    "<evidence>",
    "<final>",
]


@dataclass
class TinyLMConfig:
    """Architecture hyper-parameters for TinyLM-50.

    Defaults reproduce the ~49M parameter target:
      16k x 512 embedding (tied) + 14 blocks of GQA(8 heads / 2 KV heads,
      head_dim 64) + SwiGLU FFN(1472).
    """

    vocab_size: int = 16000
    hidden_size: int = 512
    num_layers: int = 14
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    ffn_hidden_size: int = 1472
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    initializer_range: float = 0.02

    pad_token_id: int = 0
    unk_token_id: int = 15999
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Training-time behaviour that changes the forward graph.
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_heads*head_dim ({self.num_heads * self.head_dim})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        for name in ("vocab_size", "hidden_size", "num_layers", "max_seq_len", "ffn_hidden_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TinyLMConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path, section: str = "model") -> "TinyLMConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if section and section in raw:
            raw = raw[section]
        return cls.from_dict(dict(raw))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "TinyLMConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------- parameter math
    def attention_parameters(self) -> int:
        q = self.hidden_size * self.num_heads * self.head_dim
        kv = self.hidden_size * self.num_kv_heads * self.head_dim
        out = self.num_heads * self.head_dim * self.hidden_size
        return q + 2 * kv + out

    def ffn_parameters(self) -> int:
        # gate_proj + up_proj + down_proj, no biases
        return 3 * self.hidden_size * self.ffn_hidden_size

    def block_parameters(self) -> int:
        norms = 2 * self.hidden_size
        return self.attention_parameters() + self.ffn_parameters() + norms

    def estimate_parameters(self) -> Dict[str, int]:
        """Analytic parameter budget (verified against the real module in tests)."""
        embedding = self.vocab_size * self.hidden_size
        blocks = self.num_layers * self.block_parameters()
        final_norm = self.hidden_size
        head = 0 if self.tie_embeddings else self.vocab_size * self.hidden_size
        total = embedding + blocks + final_norm + head
        return {
            "embedding": embedding,
            "blocks": blocks,
            "final_norm": final_norm,
            "lm_head": head,
            "total": total,
            "non_embedding": total - embedding,
        }


def load_config(path: Optional[str | Path] = None) -> TinyLMConfig:
    if path is None:
        return TinyLMConfig()
    return TinyLMConfig.from_yaml(path)
