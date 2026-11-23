"""Post-training weight-only quantisation (INT8, and experimental INT4).

Only ``nn.Linear`` weights are quantised: embeddings and the tied LM head stay
in float32, which is standard for weight-only quantisation of small models and
avoids duplicating the tied table. Weights are quantised per output channel
(symmetric, no zero point); activations remain float32 and the weight is
dequantised inside ``forward``. That is slower than a fused kernel, so the win
measured here is checkpoint size, not CPU speed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import TinyLMConfig
from model.model import TinyLM
from training.common import load_checkpoint


@dataclass
class QuantizationReport:
    bits: int
    quantized_linears: int
    quantized_parameters: int
    original_bytes: int
    quantized_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / self.quantized_bytes if self.quantized_bytes else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "bits": self.bits,
            "quantized_linears": self.quantized_linears,
            "quantized_parameters": self.quantized_parameters,
            "original_weight_bytes": self.original_bytes,
            "quantized_weight_bytes": self.quantized_bytes,
            "compression_ratio": self.compression_ratio,
        }
        return payload


class QuantizedLinear(nn.Module):
    """Weight-only symmetric quantisation with per-output-channel scales."""

    def __init__(self, in_features: int, out_features: int, bits: int = 8, bias: bool = False) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        if bits == 8:
            self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
        else:
            if in_features % 2 != 0:
                raise ValueError("INT4 requires an even number of input features")
            self.register_buffer(
                "weight_q",
                torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8) -> "QuantizedLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            bias=linear.bias is not None,
        )
        module.quantize_(linear.weight.detach())
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.detach())
        return module

    # -------------------------------------------------------------- quantise
    @torch.no_grad()
    def quantize_(self, weight: torch.Tensor) -> None:
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"weight shape {tuple(weight.shape)} does not match module")
        weight = weight.float()
        if self.bits == 8:
            qmax = 127.0
            scales = weight.abs().amax(dim=1).clamp(min=1e-8) / qmax
            q = torch.round(weight / scales[:, None]).clamp(-qmax, qmax).to(torch.int8)
            self.weight_q.copy_(q)
        else:
            qmax, qmin = 7.0, -8.0
            scales = weight.abs().amax(dim=1).clamp(min=1e-8) / qmax
            q = torch.round(weight / scales[:, None]).clamp(qmin, qmax).to(torch.int8)
            low = (q[:, 0::2] & 0x0F).to(torch.uint8)
            high = ((q[:, 1::2] & 0x0F) << 4).to(torch.uint8)
            self.weight_q.copy_(low | high)
        self.scales.copy_(scales)

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        if self.bits == 8:
            return self.weight_q.to(torch.float32) * self.scales[:, None]
        packed = self.weight_q
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        interleaved = torch.stack((low, high), dim=-1).reshape(self.out_features, self.in_features)
        return interleaved.to(torch.float32) * self.scales[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize_weight().to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, bias={self.bias is not None}"
        )


def quantize_model(model: TinyLM, bits: int = 8, skip_tied_head: bool = True) -> Tuple[TinyLM, QuantizationReport]:
    """Replace eligible ``nn.Linear`` modules in place and report size math."""
    original_bytes = 0
    quantized_bytes = 0
    count = 0
    parameters = 0

    def replace(parent: nn.Module, prefix: str = "") -> None:
        nonlocal original_bytes, quantized_bytes, count, parameters
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if skip_tied_head and qualified == "lm_head":
                    continue
                quantized = QuantizedLinear.from_linear(child, bits=bits)
                original_bytes += child.weight.numel() * 4
                quantized_bytes += quantized.weight_q.numel() * quantized.weight_q.element_size()
                quantized_bytes += quantized.scales.numel() * 4
                if quantized.bias is not None:
                    original_bytes += quantized.bias.numel() * 4
                    quantized_bytes += quantized.bias.numel() * 4
                parameters += child.weight.numel()
                count += 1
                setattr(parent, name, quantized)
            else:
                replace(child, qualified)

    replace(model)
    model.to(torch.float32)
    return model, QuantizationReport(
        bits=bits,
        quantized_linears=count,
        quantized_parameters=parameters,
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
    )


def load_quantized(path: str | Path) -> Tuple[TinyLM, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "tinylm-quantized":
        raise ValueError(f"{path} is not a quantized TinyLM checkpoint")
    config = TinyLMConfig.from_dict(payload["model_config"])
    model = TinyLM(config)
    model, _ = quantize_model(model, bits=int(payload["quantization"]["bits"]))
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, payload


def save_quantized(path: str | Path, model: TinyLM, config: TinyLMConfig, report: QuantizationReport, step: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "tinylm-quantized",
        "step": step,
        "model_config": config.to_dict(),
        "quantization": report.as_dict(),
        "model": model.state_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path
