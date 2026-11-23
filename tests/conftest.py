from __future__ import annotations

from pathlib import Path

import pytest

from data.pack import pack_documents
from model.config import TinyLMConfig
from model.tokenizer import TinyTokenizer

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "datasets" / "documents"


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory) -> TinyTokenizer:
    """A small, real SentencePiece model trained once for the test session."""
    directory = tmp_path_factory.mktemp("tokenizer")
    corpus = directory / "corpus.txt"
    parts = []
    for path in sorted(DOCUMENTS.glob("*.md")):
        parts.append(path.read_text(encoding="utf-8"))
    parts.append(
        "The quick brown fox jumps over the lazy dog. "
        "0123456789 alpha beta gamma delta epsilon zeta eta theta iota kappa. "
        "def add(a, b):\n    return a + b\n"
        "Question: what is a token? Answer: a unit of text. "
        "SELECT id, name FROM documents WHERE id = 7; "
    )
    corpus.write_text("\n\n".join(parts) * 3, encoding="utf-8")
    return TinyTokenizer.train(
        [corpus],
        model_prefix=directory / "tiny_test",
        vocab_size=500,
        character_coverage=0.9995,
        hard_vocab_limit=False,
    )


@pytest.fixture
def tiny_config(tokenizer: TinyTokenizer) -> TinyLMConfig:
    config = TinyLMConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ffn_hidden_size=64,
        max_seq_len=64,
    )
    return tokenizer.apply_to_config(config)


@pytest.fixture
def tiny_model(tiny_config: TinyLMConfig):
    from model.model import TinyLM

    model = TinyLM(tiny_config)
    model.eval()
    return model


@pytest.fixture
def packed_data(tmp_path, tokenizer: TinyTokenizer):
    sentence = "the quick brown fox jumps over the lazy dog and then runs away quickly. "
    documents = [(f"doc{i}", sentence * 20) for i in range(40)]
    train = pack_documents(
        documents[:32], tokenizer, tmp_path / "train", seq_len=16
    )
    valid = pack_documents(documents[32:], tokenizer, tmp_path / "valid", seq_len=16)
    return train, valid
