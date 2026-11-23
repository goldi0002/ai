from __future__ import annotations

from model.config import SPECIAL_TOKENS, TinyLMConfig


def test_special_token_ids_are_fixed(tokenizer):
    ids = tokenizer.special_ids
    assert ids["<pad>"] == 0
    assert ids["<unk>"] == 1
    assert ids["<bos>"] == 2
    assert ids["<eos>"] == 3
    assert len(set(ids.values())) == len(SPECIAL_TOKENS)
    assert set(ids) == set(SPECIAL_TOKENS)


def test_roundtrip_ascii(tokenizer):
    text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(text)
    assert ids
    assert tokenizer.decode(ids) == text


def test_roundtrip_unicode(tokenizer):
    text = "héllo 世界 — naïve café"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_empty_string(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_bos_and_eos_flags(tokenizer):
    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.special_ids["<bos>"]
    assert ids[-1] == tokenizer.special_ids["<eos>"]


def test_special_tokens_survive_roundtrip(tokenizer):
    text = "<bos><user>hi<tool_call>{\"name\": \"calculator\"}<final>ok<eos>"
    ids = tokenizer.encode(text)
    assert tokenizer.id("<tool_call>") in ids
    assert tokenizer.decode(ids) == text
    decoded_without = tokenizer.decode(ids, skip_special_tokens=True)
    assert "<tool_call>" not in decoded_without


def test_apply_to_config(tokenizer):
    config = TinyLMConfig(vocab_size=1000)
    tokenizer.apply_to_config(config)
    assert config.vocab_size == tokenizer.vocab_size
    assert config.pad_token_id == 0
    assert config.unk_token_id == 1
    assert config.eos_token_id == 3


def test_metadata_file_matches(tokenizer):
    meta = tokenizer.load_meta(tokenizer.model_path)
    assert meta["vocab_size"] == tokenizer.vocab_size
    assert meta["special_ids"] == tokenizer.special_ids
