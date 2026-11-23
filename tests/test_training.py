from __future__ import annotations

import json
from pathlib import Path

import torch

from model.config import TinyLMConfig
from model.model import TinyLM
from training.common import cosine_lr, load_checkpoint, save_checkpoint
from training.data import (
    PretrainSource,
    conversation_to_example,
    pad_batch,
    render_conversation,
)
from training.trainer import Trainer, TrainingConfig


def _config(tokenizer) -> TinyLMConfig:
    config = TinyLMConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ffn_hidden_size=64,
        max_seq_len=32,
    )
    return tokenizer.apply_to_config(config)


def test_cosine_schedule_endpoints():
    assert cosine_lr(0, 100, 1e-3, warmup_ratio=0.1) == 1e-4
    assert abs(cosine_lr(9, 100, 1e-3, warmup_ratio=0.1) - 1e-3) < 1e-12
    final = cosine_lr(99, 100, 1e-3, warmup_ratio=0.1, min_lr_ratio=0.1)
    assert abs(final - 1e-4) < 1e-5


def test_training_reduces_loss_on_fixed_data(tokenizer, packed_data, tmp_path):
    train, valid = packed_data
    config = _config(tokenizer)
    torch.manual_seed(0)
    model = TinyLM(config)
    source = PretrainSource(train, valid, seed=1)
    train_cfg = TrainingConfig(
        batch_size=2,
        grad_accum_steps=1,
        max_steps=40,
        warmup_ratio=0.1,
        learning_rate=3e-3,
        min_lr_ratio=0.2,
        weight_decay=0.0,
        eval_every=0,
        checkpoint_every=0,
        log_every=10,
        checkpoint_dir=str(tmp_path / "ckpt"),
        gradient_checkpointing=False,
    )
    trainer = Trainer(model, source, train_cfg, device=torch.device("cpu"), dtype=torch.float32)
    summary = trainer.train()
    assert summary["steps"] == 40
    assert summary["tokens_seen"] > 0
    assert summary["first_train_loss"] > summary["final_train_loss"]
    assert summary["final_train_loss"] < summary["first_train_loss"] * 0.8
    log_path = tmp_path / "ckpt" / "train_log.jsonl"
    assert log_path.exists()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    loss_records = [record for record in records if "train_loss" in record]
    assert loss_records
    assert all(record["train_loss"] == record["train_loss"] for record in loss_records)
    assert loss_records[-1]["train_loss"] < loss_records[0]["train_loss"]


def test_checkpoint_roundtrip_and_resume(tokenizer, packed_data, tmp_path):
    train, valid = packed_data
    config = _config(tokenizer)
    torch.manual_seed(3)
    model = TinyLM(config)
    source = PretrainSource(train, valid, seed=2)
    train_cfg = TrainingConfig(
        batch_size=2,
        grad_accum_steps=1,
        max_steps=4,
        learning_rate=1e-3,
        eval_every=0,
        checkpoint_every=0,
        log_every=100,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = Trainer(model, source, train_cfg, device=torch.device("cpu"), dtype=torch.float32)
    trainer.train()
    checkpoint_path = tmp_path / "ckpt" / "latest.pt"
    payload = load_checkpoint(checkpoint_path)
    assert payload["step"] == 4
    assert payload["model_config"]["hidden_size"] == 32
    assert payload["tokenizer_meta"]["train_sequences"] == len(train)

    restored = TinyLM(TinyLMConfig.from_dict(payload["model_config"]))
    restored.load_state_dict(payload["model"])
    for (name, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert name in model.state_dict()
        assert torch.equal(a, b)

    # Resume for two more steps and verify the step counter and weights move.
    resumed_cfg = TrainingConfig(**{**train_cfg.to_dict(), "max_steps": 6})
    trainer2 = Trainer(restored, source, resumed_cfg, device=torch.device("cpu"), dtype=torch.float32)
    trainer2.load_optimizer_state(payload["optimizer"])
    before = {name: tensor.clone() for name, tensor in restored.state_dict().items()}
    summary = trainer2.train(resume_step=payload["step"], tokens_seen=payload["tokens_seen"])
    assert summary["steps"] == 6
    assert any(
        not torch.equal(before[name], tensor) for name, tensor in restored.state_dict().items()
    )


def test_conversation_rendering_masks_non_assistant(tokenizer):
    messages = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "calculator", "arguments": {"expression": "2+2"}}],
        },
        {"role": "tool", "content": '{"ok": true, "content": "2+2 = 4"}'},
        {"role": "assistant", "content": "<final>The answer is 4."},
    ]
    ids, mask = render_conversation(tokenizer, messages)
    assert ids[0] == tokenizer.id("<bos>")
    assert tokenizer.id("<tool_call>") in ids
    assistant_positions = [i for i, flag in enumerate(mask) if flag]
    first_assistant = ids.index(tokenizer.id("<assistant>"))
    assert min(assistant_positions) == first_assistant
    # Every masked target is inside an assistant span.
    assert not mask[ids.index(tokenizer.id("<user>"))]

    input_ids, labels = conversation_to_example(tokenizer, messages)
    assert len(input_ids) == len(labels)
    # labels[i] trains the token at i+1, so the last pre-assistant position
    # legitimately predicts the <assistant> role token itself.
    assert all(label == -100 for label in labels[: first_assistant - 1])
    assert labels[first_assistant - 1] == tokenizer.id("<assistant>")
    assert any(label != -100 for label in labels)


def test_pad_batch_shapes(tokenizer):
    examples = [([1, 2, 3], [-100, 2, 3]), ([4, 5], [-100, 5])]
    input_ids, labels = pad_batch(examples, tokenizer.special_ids["<pad>"])
    assert input_ids.shape == labels.shape == (2, 3)
    assert input_ids[1, 2] == tokenizer.special_ids["<pad>"]
    assert labels[1, 2] == -100


def test_save_checkpoint_is_atomic_and_self_describing(tmp_path, tiny_model, tiny_config):
    path = tmp_path / "ckpt" / "a.pt"
    save_checkpoint(
        path,
        tiny_model,
        None,
        step=7,
        model_config=tiny_config,
        train_config={"run": "unit"},
        tokenizer_meta={"vocab_size": tiny_config.vocab_size},
        tokens_seen=123,
    )
    assert path.exists() and not (tmp_path / "ckpt" / "a.pt.tmp").exists()
    payload = load_checkpoint(path)
    assert payload["step"] == 7 and payload["tokens_seen"] == 123
    assert payload["model_config"]["vocab_size"] == tiny_config.vocab_size
