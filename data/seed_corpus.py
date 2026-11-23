"""Build a deterministic synthetic seed corpus.

WHY THIS EXISTS
---------------
Tokenizers need a few megabytes of text to learn a 16,000-piece BPE vocabulary,
and smoke-testing the training loop needs data that ships with the repository.
This module expands hand-authored templates into varied, well-formed technical
prose, arithmetic walkthroughs, Python examples and dialogues.

WHAT THIS IS NOT
----------------
This is **not** a pretraining corpus and must not be presented as one. It is
synthetic, repetitive and small. Real pretraining requires licensed corpora
(see ``datasets/sources.yaml`` and ``python -m data.fetch --help``). The
dataset manifest records ``synthetic: true`` for every file produced here.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

CATEGORY_SHARE = {
    "educational": 0.20,
    "prose": 0.15,
    "technical": 0.13,
    "programming": 0.13,
    "qa": 0.09,
    "math": 0.08,
    "conversation": 0.04,
    "identifiers": 0.18,  # subword variety for a 16k BPE vocabulary
}

TERMS = [
    ("tensor", "a multidimensional array of numbers with a fixed shape and dtype"),
    ("gradient", "the vector of partial derivatives of a loss with respect to its inputs"),
    ("learning rate", "the step size used when updating parameters during optimisation"),
    ("epoch", "one complete pass over the training dataset"),
    ("batch", "a group of examples processed together in one forward pass"),
    ("attention", "a weighted lookup where each query selects relevant key-value pairs"),
    ("embedding", "a dense vector representation learned for a discrete token"),
    ("residual connection", "an identity path that adds a block's input to its output"),
    ("normalisation", "rescaling activations so that optimisation remains stable"),
    ("token", "the atomic unit of text handled by a language model"),
    ("perplexity", "the exponentiated average negative log-likelihood of a corpus"),
    ("quantisation", "storing weights with fewer bits to reduce memory and latency"),
    ("index", "a data structure that speeds up lookups at the cost of extra storage"),
    ("transaction", "a group of database operations that either all commit or all roll back"),
    ("deadlock", "a cycle of processes each waiting for a resource the others hold"),
    ("cache", "a small fast store that keeps recently used data close to the consumer"),
    ("latency", "the time between a request and its response"),
    ("throughput", "the amount of work completed per unit of time"),
    ("entropy", "the average surprise of a probability distribution"),
    ("softmax", "a function that turns arbitrary scores into a probability distribution"),
    ("logit", "an unnormalised score produced before the softmax layer"),
    ("precision", "the numeric format used to store and compute with values"),
    ("checkpoint", "a saved snapshot of weights, optimiser state and training progress"),
    ("corpus", "the collection of documents used to train or evaluate a system"),
    ("inference", "running a trained model to produce predictions"),
    ("tokenizer", "the component that maps text to token ids and back"),
    ("shard", "one of several files that together hold a large dataset"),
    ("manifest", "a machine-readable record describing the contents of a dataset"),
    ("citation", "a reference that ties a claim to a specific source passage"),
    ("provenance", "the origin and processing history of a piece of data"),
]

ALGORITHMS = [
    (
        "binary search",
        [
            "Keep two indices low and high that bound the search interval.",
            "While low is less than or equal to high, compute the midpoint.",
            "Compare the midpoint value with the target.",
            "If they match, return the midpoint index.",
            "If the midpoint value is too small, move low past the midpoint.",
            "Otherwise move high before the midpoint.",
            "If the loop ends, the target is absent and the search returns minus one.",
        ],
    ),
    (
        "gradient descent",
        [
            "Initialise the parameters with small random values.",
            "Compute the loss for a batch of examples.",
            "Backpropagate to obtain the gradient of the loss.",
            "Multiply the gradient by the learning rate to get the step size.",
            "Subtract the scaled gradient from the parameters.",
            "Repeat until the validation loss stops improving.",
        ],
    ),
    (
        "deduplication",
        [
            "Normalise each document so that whitespace differences disappear.",
            "Hash the normalised text to detect exact duplicates immediately.",
            "Build word shingles and compress them into a compact signature.",
            "Group signatures into bands so that similar documents collide.",
            "Compare candidates inside each bucket and drop near duplicates.",
            "Record how many documents were removed for the dataset manifest.",
        ],
    ),
    (
        "full text search",
        [
            "Split each document into tokens and normalise their case.",
            "Insert the tokens into an inverted index keyed by term.",
            "Store term frequencies and document lengths for ranking.",
            "At query time, look up posting lists for each query term.",
            "Score candidates with a ranking function such as BM25.",
            "Return the highest scoring chunks together with their source ids.",
        ],
    ),
]

PROSE_OPENERS = [
    "A practical system is usually built from a few simple parts that agree on their contracts.",
    "When a program grows, clarity becomes more valuable than cleverness.",
    "Most engineering decisions trade one kind of cost for another.",
    "Reliable software is the result of small, testable steps rather than one large leap.",
    "Understanding a system means knowing which parts can fail and how that failure is contained.",
]

PROSE_BODIES = [
    "Readers benefit from concrete examples, measured numbers and explicit assumptions. A claim without a measurement is a hypothesis, and a hypothesis should be labelled as such.",
    "The cheapest debugging tool is a clear error message that names the component, the input and the expectation that was violated.",
    "Documentation that explains why a decision was made ages better than documentation that only restates the code.",
    "Performance work should start with a profile, because intuition about hot paths is often wrong.",
    "Good interfaces hide accidental complexity while exposing the decisions a caller genuinely needs to make.",
]

PY_TEMPLATES = [
    (
        "mean",
        "def mean(values):\n    \"\"\"Return the arithmetic mean of a non-empty sequence.\"\"\"\n    if not values:\n        raise ValueError(\"mean of empty sequence\")\n    return sum(values) / len(values)\n",
        "print(mean([1, 2, 3, 4]))  # 2.5",
    ),
    (
        "clamp",
        "def clamp(value, low, high):\n    \"\"\"Limit value to the inclusive range [low, high].\"\"\"\n    return max(low, min(value, high))\n",
        "print(clamp(12, 0, 10))  # 10",
    ),
    (
        "chunk",
        "def chunk(items, size):\n    \"\"\"Yield consecutive slices of at most size items.\"\"\"\n    for start in range(0, len(items), size):\n        yield items[start : start + size]\n",
        "print(list(chunk([1, 2, 3, 4, 5], 2)))  # [[1, 2], [3, 4], [5]]",
    ),
    (
        "safe_div",
        "def safe_div(a, b, default=0.0):\n    \"\"\"Divide a by b, returning default when b is zero.\"\"\"\n    if b == 0:\n        return default\n    return a / b\n",
        "print(safe_div(1, 0))  # 0.0",
    ),
    (
        "count_words",
        "def count_words(text):\n    \"\"\"Count word occurrences, ignoring case.\"\"\"\n    counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n",
        "print(count_words(\"the cat and the hat\"))  # {'the': 2, 'cat': 1, ...}",
    ),
    (
        "cosine_similarity",
        "def cosine_similarity(a, b):\n    \"\"\"Cosine of the angle between two equal-length vectors.\"\"\"\n    if len(a) != len(b):\n        raise ValueError(\"vectors must have equal length\")\n    dot = sum(x * y for x, y in zip(a, b))\n    norm_a = sum(x * x for x in a) ** 0.5\n    norm_b = sum(y * y for y in b) ** 0.5\n    if norm_a == 0 or norm_b == 0:\n        return 0.0\n    return dot / (norm_a * norm_b)\n",
        "print(round(cosine_similarity([1, 0], [1, 1]), 3))  # 0.707",
    ),
    (
        "moving_average",
        "def moving_average(values, window):\n    \"\"\"Return a list of trailing averages with the given window size.\"\"\"\n    if window <= 0:\n        raise ValueError(\"window must be positive\")\n    out = []\n    total = 0.0\n    for index, value in enumerate(values):\n        total += value\n        if index >= window:\n            total -= values[index - window]\n        if index >= window - 1:\n            out.append(total / window)\n    return out\n",
        "print(moving_average([1, 2, 3, 4], 2))  # [1.5, 2.5, 3.5]",
    ),
]


def _glossary(rng: random.Random, count: int) -> List[str]:
    paragraphs = []
    for _ in range(count):
        term, definition = rng.choice(TERMS)
        paragraphs.append(
            f"{term.capitalize()} is {definition}. In practice, {term} matters because it "
            "determines how the surrounding system behaves under load and how easily it can "
            "be observed when something goes wrong."
        )
    return ["Glossary of machine learning and systems terms.", "", *paragraphs]


def _prose(rng: random.Random, count: int) -> List[str]:
    paragraphs = []
    for _ in range(count):
        paragraphs.append(" ".join([rng.choice(PROSE_OPENERS), rng.choice(PROSE_BODIES), rng.choice(PROSE_BODIES)]))
    return ["Notes on building reliable software.", "", *paragraphs]


def _algorithms(rng: random.Random, count: int) -> List[str]:
    sections = []
    for _ in range(count):
        name, steps = rng.choice(ALGORITHMS)
        sections.append(f"Algorithm: {name}")
        for number, step in enumerate(steps, start=1):
            sections.append(f"Step {number}. {step}")
        sections.append(
            "Each step should be testable on its own, which is why the description keeps the "
            "state visible instead of hiding it inside one large operation."
        )
        sections.append("")
    return sections


def _math(rng: random.Random, count: int) -> List[str]:
    paragraphs = []
    for _ in range(count):
        a = rng.randint(2, 60)
        b = rng.randint(2, 40)
        c = rng.randint(1, 9)
        total = a * b + c
        difference = total - c
        paragraphs.append(
            f"Worked example. Multiply {a} by {b} to get {a * b}. Adding {c} gives {total}. "
            f"Subtracting {c} again returns {difference}, which confirms the arithmetic. "
            f"The average of {a} and {b} is {(a + b) / 2:.2f}, and the ratio {a}/{b} is "
            f"{a / b:.4f}."
        )
        paragraphs.append(
            f"If a quantity grows by {c} percent each period, after two periods it is "
            f"multiplied by {(1 + c / 100) ** 2:.4f}. Growth compounds because the second "
            "period applies to the already increased value."
        )
    return ["Mathematics and logic exercises.", "", *paragraphs]


def _programming(rng: random.Random, count: int) -> List[str]:
    sections = []
    for _ in range(count):
        name, code, usage = rng.choice(PY_TEMPLATES)
        sections.append(f"Example: {name}")
        sections.append("```python")
        sections.append(code.rstrip())
        sections.append("```")
        sections.append(f"Usage: {usage}")
        sections.append(
            "The function validates its inputs, keeps a single responsibility and returns a "
            "predictable type, which makes it easy to test."
        )
        sections.append("")
    return sections


def _qa(rng: random.Random, count: int) -> List[str]:
    pairs = []
    for _ in range(count):
        term, definition = rng.choice(TERMS)
        pairs.append(f"Question: What is {term}?\nAnswer: {term.capitalize()} is {definition}.")
        name, steps = rng.choice(ALGORITHMS)
        pairs.append(
            f"Question: How does {name} work?\nAnswer: {steps[0]} {steps[1]} {steps[-1]}"
        )
    return ["Question and answer notes.", "", *pairs]


SYLLABLES_A = [
    "arc", "bel", "cor", "dun", "eth", "fal", "gar", "hel", "ion", "jor",
    "kel", "lum", "mor", "nax", "orb", "pyr", "quel", "rho", "syl", "tor",
    "umb", "vex", "wyn", "xan", "yar", "zel", "brim", "clast", "drav", "fenn",
]
SYLLABLES_B = [
    "ander", "anth", "ora", "ell", "ith", "ova", "une", "aris", "eton", "ika",
    "olis", "ura", "eron", "ista", "aro", "yne", "adis", "ento", "ium", "ax",
]
SERVICE_PARTS = [
    "gateway", "scheduler", "ingester", "indexer", "renderer", "collector",
    "resolver", "dispatcher", "validator", "compactor", "forwarder", "registry",
]


def _identifiers(rng: random.Random, count: int) -> List[str]:
    """Generated identifiers, paths and log lines.

    A 16,000-piece BPE vocabulary needs more distinct substrings than
    hand-written prose of this size can supply. This section is synthetic and
    labelled as such; it exists to give the tokenizer varied technical
    subwords, not to teach the model facts.
    """
    lines: List[str] = []
    for index in range(max(200, count)):
        word = rng.choice(SYLLABLES_A) + rng.choice(SYLLABLES_B) + str(rng.randint(0, 999))
        other = rng.choice(SYLLABLES_A) + rng.choice(SYLLABLES_B)
        service = rng.choice(SERVICE_PARTS) + "-" + str(rng.randint(1, 40))
        trace = "".join(rng.choice("0123456789abcdef") for _ in range(12))
        path = f"/srv/{service.replace('-', '_')}/{word}/{other}_{rng.randint(1, 9)}.jsonl"
        lines.append(
            f"trace_id={trace} service={service} span={word} parent={other} "
            f"status={rng.choice([200, 201, 204, 400, 404, 409, 429, 500, 503])} "
            f"latency_ms={rng.randint(1, 4000)} retries={rng.randint(0, 5)} path={path}"
        )
        if index % 6 == 0:
            lines.append(
                f"def {word}({other}_id: int, {other}_limit: int = {rng.randint(1, 500)}) -> dict:\n"
                f"    \"\"\"Return the {other} record for {word}.\"\"\"\n"
                f"    return {{'{other}_id': {other}_id, 'limit': {other}_limit}}"
            )
        if index % 9 == 0:
            lines.append(
                f"Question: How do I configure {word} in {service}?\n"
                f"Answer: Set the {other} option in {path} and reload {service}."
            )
        if index % 13 == 0:
            lines.append(
                f"Table {word}\n"
                f"| field | type | note |\n| --- | --- | --- |\n"
                f"| {other}_id | integer | primary key |\n"
                f"| {word}_state | text | one of active, paused, archived |"
            )
    return ["Generated identifiers and log records (synthetic).", "", *lines]


DIVERSITY_A = [
    "arc", "bel", "cor", "dun", "eth", "fal", "gar", "hel", "ion", "jor",
    "kel", "lum", "mor", "nax", "orb", "pyr", "quel", "rho", "syl", "tor",
    "umb", "vex", "wyn", "xan", "yar", "zel", "brim", "clast", "drav", "fenn",
]
DIVERSITY_B = [
    "ander", "anth", "ora", "ell", "ith", "ova", "une", "aris", "eton", "ika",
    "olis", "ura", "eron", "ista", "aro", "yne", "adis", "ento", "ium", "ax",
]
DIVERSITY_C = [
    "ix", "on", "ar", "em", "ul", "is", "ek", "ov", "an", "ir",
    "us", "al", "en", "or", "im", "ur", "as", "et", "id", "ok",
]


def build_diversity_corpus(lines: int = 30_000, seed: int = 2024) -> str:
    """Deterministic identifier/log corpus used **only** to train the tokenizer.

    A 16k-piece BPE vocabulary needs far more distinct substrings than the
    hand-written seed corpus contains. This generated text supplies that
    variety (identifiers, paths, log fields). It is never used for
    pretraining, and every generated tokenizer records that distinction.
    """
    rng = random.Random(seed)
    output = [
        "# SYNTHETIC TOKENIZER DIVERSITY CORPUS (generated by data/seed_corpus.py)\n"
        "# Used only to train subword pieces. Never used as pretraining text.\n"
    ]
    services = [
        "gateway", "scheduler", "ingester", "indexer", "renderer", "collector",
        "resolver", "dispatcher", "validator", "compactor", "forwarder", "registry",
    ]
    for index in range(lines):
        word = (
            rng.choice(DIVERSITY_A)
            + rng.choice(DIVERSITY_B)
            + rng.choice(DIVERSITY_C)
            + str(rng.randint(0, 99))
        )
        other = rng.choice(DIVERSITY_A) + rng.choice(DIVERSITY_C)
        service = rng.choice(services) + "-" + str(rng.randint(1, 40))
        output.append(
            f"trace_id={index:08x} service={service} span={word} parent={other} "
            f"status={rng.choice([200, 201, 204, 400, 404, 409, 429, 500, 503])} "
            f"latency_ms={rng.randint(1, 4000)} retries={rng.randint(0, 5)} "
            f"path=/srv/{service.replace('-', '_')}/{other}/{word}.jsonl"
        )
    return "\n".join(output) + "\n"


def write_diversity_corpus(path: str | Path, lines: int = 30_000, seed: int = 2024) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_diversity_corpus(lines=lines, seed=seed), encoding="utf-8")
    return path


def _conversation(rng: random.Random, count: int) -> List[str]:
    lines = []
    for _ in range(count):
        term, definition = rng.choice(TERMS)
        lines.append(f"User: Can you explain {term} in one sentence?")
        lines.append(f"Assistant: {term.capitalize()} is {definition}.")
        lines.append("User: Thanks. What should I watch out for?")
        lines.append(
            "Assistant: Measure before optimising, validate inputs at the boundary, and write "
            "down the assumption you are relying on."
        )
        lines.append("")
    return lines


BUILDERS = {
    "educational": _algorithms,
    "prose": _prose,
    "technical": _glossary,
    "programming": _programming,
    "qa": _qa,
    "math": _math,
    "conversation": _conversation,
    "identifiers": _identifiers,
}


def build_corpus(target_kb: int = 1500, seed: int = 1337) -> dict[str, str]:
    """Return {category: text} totalling roughly ``target_kb`` kilobytes."""
    rng = random.Random(seed)
    total_chars = max(target_kb, 50) * 1024
    output: dict[str, str] = {}
    for category, share in CATEGORY_SHARE.items():
        wanted = int(total_chars * share)
        builder = BUILDERS[category]
        # Each paragraph block is ~300-600 chars; overshoot then trim on a line boundary.
        blocks_needed = max(8, wanted // 400)
        text = "\n".join(builder(rng, blocks_needed))
        if len(text) > wanted:
            text = text[:wanted].rsplit("\n", 1)[0]
        header = (
            "# SYNTHETIC SEED CORPUS (generated by data/seed_corpus.py)\n"
            f"# category: {category}\n"
            "# Not a pretraining corpus. Use only for tokenizer training and smoke tests.\n\n"
        )
        output[category] = header + text + "\n"
    return output


def write_corpus(out_dir: str | Path, target_kb: int = 1500, seed: int = 1337) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for category, text in build_corpus(target_kb=target_kb, seed=seed).items():
        path = out_dir / f"synthetic_{category}.txt"
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="datasets/pretraining/seed_generated")
    parser.add_argument("--target-kb", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    paths = write_corpus(args.out_dir, target_kb=args.target_kb, seed=args.seed)
    total = sum(path.stat().st_size for path in paths)
    for path in paths:
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")
    print(f"total {total / 1024:.1f} KiB synthetic seed text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
