from __future__ import annotations

import pytest
import torch

from model.config import TinyLMConfig
from model.model import TinyLM


def test_default_config_parameter_count_matches_analytic():
    config = TinyLMConfig()
    model = TinyLM(config)
    analytic = config.estimate_parameters()
    measured = model.parameter_breakdown()
    assert measured == analytic
    assert measured["total"] == model.num_parameters()
    # The documented target is ~49M; fail loudly if the budget drifts.
    assert 48_500_000 <= measured["total"] <= 49_500_000
    assert measured["total"] == 49_035_776


def test_forward_shapes_and_no_loss(tiny_model, tiny_config):
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = tiny_model(input_ids)
    assert out.logits.shape == (2, 16, tiny_config.vocab_size)
    assert out.loss is None
    assert out.past_key_values is None


def test_forward_loss_and_backward(tiny_model, tiny_config):
    tiny_model.train()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 12))
    targets = input_ids.clone()
    targets[:, 0] = -100
    out = tiny_model(input_ids, targets=targets)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_causality_prefix_unchanged(tiny_model, tiny_config):
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 20))
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % tiny_config.vocab_size
    with torch.no_grad():
        a = tiny_model(ids).logits
        b = tiny_model(changed).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_kv_cache_matches_full_forward(tiny_model, tiny_config):
    torch.manual_seed(1)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 12))
    with torch.no_grad():
        full = tiny_model(ids).logits
        prefix = tiny_model(ids[:, :8], use_cache=True)
        step = tiny_model(ids[:, 8:], past_key_values=prefix.past_key_values, use_cache=True)
    assert len(prefix.past_key_values) == tiny_config.num_layers
    assert torch.allclose(full[:, -1], step.logits[:, -1], atol=1e-4)


def test_generation_extends_prompt_deterministically(tiny_model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (1, 5))
    first = tiny_model.generate(ids, max_new_tokens=6, temperature=0.0)
    second = tiny_model.generate(ids, max_new_tokens=6, temperature=0.0)
    assert first.shape[0] == 1 and first.shape[1] >= ids.shape[1]
    assert first.shape[1] <= ids.shape[1] + 6
    assert torch.equal(first[:, : ids.shape[1]], ids)
    assert torch.equal(first, second)


def test_cached_generation_matches_uncached(tiny_model, tiny_config):
    torch.manual_seed(5)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 6))
    with_cache = tiny_model.generate(ids, max_new_tokens=8, temperature=0.0, use_cache=True)
    without_cache = tiny_model.generate(ids, max_new_tokens=8, temperature=0.0, use_cache=False)
    assert torch.equal(with_cache, without_cache)


def test_generation_stops_at_eos(tiny_model, tiny_config):
    ids = torch.tensor([[tiny_config.bos_token_id, tiny_config.eos_token_id]])
    out = tiny_model.generate(ids, max_new_tokens=5, temperature=0.0, eos_token_id=tiny_config.eos_token_id)
    assert out.shape[1] <= ids.shape[1] + 5


def test_invalid_inputs_rejected(tiny_model, tiny_config):
    with pytest.raises(ValueError):
        tiny_model(torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        tiny_model(torch.zeros(1, 4, dtype=torch.long), targets=torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError):
        TinyLMConfig(hidden_size=48, num_heads=8, head_dim=8)
    with pytest.raises(ValueError):
        TinyLMConfig(num_heads=8, num_kv_heads=3)
