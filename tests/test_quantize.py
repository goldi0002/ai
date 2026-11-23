from __future__ import annotations

import torch

from model.model import TinyLM
from runtime.quantize import (
    QuantizedLinear,
    load_quantized,
    quantize_model,
    save_quantized,
)


def test_int8_linear_quantises_and_dequantises(tiny_config):
    linear = torch.nn.Linear(32, 16)
    quantized = QuantizedLinear.from_linear(linear, bits=8)
    weight = quantized.dequantize_weight()
    assert weight.shape == (16, 32)
    assert quantized.weight_q.dtype == torch.int8
    error = (weight - linear.weight).abs().max().item()
    assert error < 0.05


def test_int4_linear_packs_two_values_per_byte(tiny_config):
    linear = torch.nn.Linear(32, 16)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    assert quantized.weight_q.dtype == torch.uint8
    assert quantized.weight_q.shape == (16, 16)  # half the input features
    weight = quantized.dequantize_weight()
    assert weight.shape == (16, 32)
    assert (weight - linear.weight).abs().max().item() < 0.2


def test_quantize_model_forward_and_report(tiny_model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (1, 8))
    with torch.no_grad():
        reference = tiny_model(x).logits
    model, report = quantize_model(tiny_model, bits=8)
    assert report.quantized_linears > 0
    assert report.quantized_parameters > 0
    assert report.quantized_bytes < report.original_bytes
    assert report.compression_ratio > 3.5
    with torch.no_grad():
        quantized = model(x).logits
    assert torch.isfinite(quantized).all()
    # Quantisation perturbs logits but must not change the function wildly.
    assert (reference - quantized).abs().mean() < 0.5


def test_save_and_load_quantized_roundtrip(tiny_model, tiny_config, tmp_path):
    model, report = quantize_model(tiny_model, bits=8)
    x = torch.randint(0, tiny_config.vocab_size, (1, 6))
    with torch.no_grad():
        before = model(x).logits
    path = tmp_path / "q8.pt"
    save_quantized(path, model, tiny_config, report, step=3)
    loaded, payload = load_quantized(path)
    assert payload["step"] == 3
    assert payload["quantization"]["bits"] == 8
    with torch.no_grad():
        after = loaded(x).logits
    assert torch.allclose(before, after, atol=1e-5)
