"""The copy curriculum must supervise only the repeated span, and repeat it exactly."""

import random

import pytest

from model.tokenizer import TinyTokenizer
from training.copy_task import CopySource

TOKENIZER = "data/tokenizer/tiny50_sp.model"


@pytest.fixture(scope="module")
def source():
    return CopySource(TinyTokenizer(TOKENIZER), max_len=64, seed=3)


def test_labels_align_with_inputs_and_mask_the_first_span(source):
    rng = random.Random(0)
    for _ in range(50):
        inputs, labels = source.example(rng)
        assert len(inputs) == len(labels)
        targets = [label for label in labels if label != -100]
        assert targets, "example has no supervised tokens"
        first = next(i for i, label in enumerate(labels) if label != -100)
        # Nothing before the separator is supervised, so the model cannot be
        # rewarded for predicting the first, unseen occurrence.
        assert source.sep in inputs[: first + 1]


def test_random_stream_repeats_exactly(source):
    shown, repeated = source._random(random.Random(5))
    assert shown == repeated
    assert all(token >= 12 for token in shown)


def test_number_stream_contains_same_numbers_in_both_spans(source):
    tok = source.tokenizer
    for seed in range(20):
        shown, repeated = source._numbers(random.Random(seed))
        digits = lambda ids: "".join(ch for ch in tok.decode(ids) if ch.isdigit() or ch == ".")
        assert digits(shown).replace(".", "") == digits(repeated).replace(".", "")


def test_batches_are_deterministic_and_padded(source):
    a = source.eval_batches(4, 2)
    b = source.eval_batches(4, 2)
    assert all((x[0] == y[0]).all() and (x[1] == y[1]).all() for x, y in zip(a, b))
    inputs, labels = a[0]
    assert inputs.shape == labels.shape and inputs.shape[0] == 4
