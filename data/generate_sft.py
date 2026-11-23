"""Generate a large, varied tool-use SFT set.

    python -m data.generate_sft --out datasets/instructions/tiny50_sft_v2.jsonl

Why this exists: the 20 hand-written conversations in ``tiny50_sft.jsonl`` are
memorised by a 49M model without generalising (24% held-out tool selection).
This script produces ~1.5k conversations with

* varied wording for the same intent,
* diverse arguments (numbers, file names, memory slots, search topics),
* negative examples (no tool needed, tool unavailable),
* multi-tool turns and tool-error turns.

Every tool result is produced by the real tool implementations, so the training
text matches the runtime format byte for byte (``ToolResult.to_json()``). Final
answers are templated from those real results or from curated facts that the
script verifies are present in the retrieved text.

Held-out safety: any prompt that equals, contains or is contained in a prompt
from ``tool_selection_eval.jsonl`` is dropped, and the search topics used by the
evaluation set are excluded. Output is deterministic for a given ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.factory import build_registry
from agent.prompts import COMPACT_SYSTEM_PROMPT
from memory.store import MemoryStore
from tools.registry import ToolRegistry

EVAL_PATH = Path("datasets/instructions/tool_selection_eval.jsonl")
HAND_WRITTEN = Path("datasets/instructions/tiny50_sft.jsonl")
DOCUMENT_ROOT = "datasets/documents"
TOOL_NAMES = "calculator, local_search, read_document, list_files, retrieve_memory and save_memory"

# Rendered training examples longer than this many tool-result characters keep
# only the tool-call turn (the model is still taught to call; the 256-token
# pretraining context cannot hold the long passage plus an answer).
MAX_RESULT_CHARS_WITH_FINAL = 1100


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# --------------------------------------------------------------------- helpers
class Builder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.tmp = tempfile.mkdtemp(prefix="tinylm_sft_")
        self.registry, self.engine, self.memory = build_registry(
            db_path=str(Path(self.tmp) / "gen.db"), document_root=DOCUMENT_ROOT
        )
        self.engine.ingest_directory(DOCUMENT_ROOT)
        self.eval_prompts = [
            _norm(json.loads(line)["prompt"])
            for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.seen_prompts: set[str] = set()
        self.out: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}

    # -- tool execution ------------------------------------------------------
    def run(self, name: str, arguments: Dict[str, Any]) -> str:
        return self.registry.execute(name, arguments).to_json()

    def fresh_memory_run(self, name: str, arguments: Dict[str, Any], store: MemoryStore) -> str:
        from tools.memory_tools import RetrieveMemory, SaveMemory

        tool = SaveMemory(store) if name == "save_memory" else RetrieveMemory(store)
        result = tool.run(**arguments)
        text = result.to_json()
        # Stable timestamps so the output is deterministic.
        stamp = "2026-%02d-%02dT%02d:%02d:00+00:00" % (
            self.rng.randint(1, 11),
            self.rng.randint(1, 28),
            self.rng.randint(0, 23),
            self.rng.randint(0, 59),
        )
        return re.sub(r"updated \d{4}-\d{2}-\d{2}T[\d:]+\+00:00", f"updated {stamp}", text)

    # -- acceptance ----------------------------------------------------------
    def accept(self, category: str, messages: List[Dict[str, Any]], target: int) -> bool:
        if self.counts.get(category, 0) >= target:
            return False
        user_prompts = [_norm(m["content"]) for m in messages if m["role"] == "user"]
        for prompt in user_prompts:
            if prompt in self.seen_prompts:
                return False
            for held in self.eval_prompts:
                if held in prompt or prompt in held:
                    return False
        self.seen_prompts.update(user_prompts)
        self.counts[category] = self.counts.get(category, 0) + 1
        self.out.append({"messages": [{"role": "system", "content": COMPACT_SYSTEM_PROMPT}] + messages})
        return True


def call(name: str, **arguments: Any) -> Dict[str, Any]:
    return {"name": name, "arguments": arguments}


def assistant_call(*calls: Dict[str, Any]) -> Dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def tool(content: str) -> Dict[str, Any]:
    return {"role": "tool", "content": content}


def final(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": f"<final>{text}"}


def user(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def fmt_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return ("%.6f" % value).rstrip("0").rstrip(".")


def result_of(tool_json: str) -> Optional[str]:
    payload = json.loads(tool_json)
    if not payload.get("ok"):
        return None
    content = payload["content"]
    return content.split(" = ", 1)[1] if " = " in content else content


# ------------------------------------------------------------------ calculator
def _calc_spec(rng: random.Random) -> Tuple[str, str, str]:
    """Return (prompt, expression, answer-sentence template using {r})."""
    kind = rng.choice(
        [
            "add", "sub", "mul", "div", "pct", "discount", "markup", "unit", "power",
            "sqrt", "avg", "chain", "price", "split", "area", "rate", "pctchange",
        ]
    )
    a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 50)
    big = rng.randint(100, 9999)
    if kind == "add":
        p = rng.choice(
            [f"What is {big} plus {a}?", f"Sum {a}, {b} and {big}.", f"Work out {big} + {b}.", f"How much is {a} added to {big}?"]
        )
        e = f"{big} + {a}" if "plus" in p or "added" in p else (f"{a} + {b} + {big}" if p.startswith("Sum") else f"{big} + {b}")
        return p, e, "The total is {r}."
    if kind == "sub":
        p = rng.choice(
            [f"What is {big} minus {a}?", f"Subtract {a} from {big}.", f"Work out {big} - {b}.", f"How much is {big} take away {a}?"]
        )
        e = f"{big} - {b}" if p.startswith("Work") else f"{big} - {a}"
        return p, e, "The difference is {r}."
    if kind == "mul":
        p = rng.choice(
            [f"What is {a} times {b}?", f"Multiply {a} by {big}.", f"Work out {a} x {b}.", f"What do you get from {b} multiplied by {c}?"]
        )
        if p.startswith("Multiply"):
            e = f"{a} * {big}"
        elif "multiplied" in p:
            e = f"{b} * {c}"
        else:
            e = f"{a} * {b}"
        return p, e, "The product is {r}."
    if kind == "div":
        p = rng.choice(
            [f"What is {big} divided by {c}?", f"Divide {big} by {a}.", f"How many times does {c} go into {big}?", f"Compute {a * c} / {c}."]
        )
        e = {0: f"{big} / {c}", 1: f"{big} / {a}", 2: f"{big} / {c}", 3: f"{a * c} / {c}"}[
            [f"What is {big} divided by {c}?", f"Divide {big} by {a}.", f"How many times does {c} go into {big}?", f"Compute {a * c} / {c}."].index(p)
        ]
        return p, e, "The result is {r}."
    if kind == "pct":
        pc = rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 60, 75])
        n = rng.choice([40, 60, 80, 120, 150, 200, 240, 300, 360, 450, 500, 640, 900])
        p = rng.choice(
            [f"What is {pc}% of {n}?", f"Find {pc} percent of {n}.", f"How much is {pc} per cent of {n}?", f"Calculate {pc}% of {n}."]
        )
        return p, f"{n} * {pc} / 100", f"{pc} percent of {n} is " + "{r}."
    if kind == "discount":
        pc = rng.choice([10, 15, 20, 25, 30, 40, 50])
        price = rng.choice([30, 45, 60, 75, 90, 120, 150, 200, 250, 320])
        item = rng.choice(["jacket", "lamp", "book", "backpack", "headset", "chair", "keyboard", "watch"])
        p = rng.choice(
            [
                f"A {item} costs {price} euros with {pc}% off. What is the sale price?",
                f"The {item} is {price} dollars and there is a {pc} percent discount. How much do I pay?",
                f"A {item} normally costs {price}. After a {pc}% reduction, what is the price?",
            ]
        )
        return p, f"{price} * (1 - {pc}/100)", "The sale price is {r}."
    if kind == "markup":
        pc = rng.choice([5, 7, 10, 19, 20, 21, 25])
        price = rng.choice([40, 80, 100, 160, 250, 400, 520])
        p = rng.choice(
            [
                f"Add {pc}% tax to {price}. What is the total?",
                f"An invoice is {price} before {pc} percent VAT. What is the total with VAT?",
                f"What is {price} after a {pc}% increase?",
            ]
        )
        return p, f"{price} * (1 + {pc}/100)", "The total is {r}."
    if kind == "unit":
        spec = rng.choice(
            [
                ("hours", "seconds", 3600), ("hours", "minutes", 60), ("minutes", "seconds", 60),
                ("days", "hours", 24), ("weeks", "days", 7), ("kilometres", "metres", 1000),
                ("kilograms", "grams", 1000), ("litres", "millilitres", 1000), ("years", "months", 12),
            ]
        )
        n = rng.choice([2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 24, 36, 2.5, 1.5, 0.5, 7.5])
        p = rng.choice(
            [
                f"How many {spec[1]} are in {n} {spec[0]}?",
                f"Convert {n} {spec[0]} to {spec[1]}.",
                f"{n} {spec[0]} is how many {spec[1]}?",
            ]
        )
        return p, f"{n} * {spec[2]}", f"{n} {spec[0]} is " + "{r}" + f" {spec[1]}."
    if kind == "power":
        base, exp = rng.randint(2, 9), rng.randint(2, 8)
        p = rng.choice(
            [f"What is {base} raised to the power of {exp}?", f"Compute {base} ^ {exp}.", f"What is {base} to the {exp}th power?"]
        )
        return p, f"{base} ** {exp}", f"{base} to the power of {exp} is " + "{r}."
    if kind == "sqrt":
        n = rng.choice([2, 3, 5, 7, 10, 50, 81, 144, 169, 200, 625, 1024])
        if rng.random() < 0.5:
            return rng.choice([f"What is the square root of {n}?", f"Find the square root of {n}."]), f"sqrt({n})", f"The square root of {n} is " + "{r}."
        d = rng.choice([1, 2, 3])
        return f"Give the square root of {n} to {d} decimal places.", f"round(sqrt({n}), {d})", f"The square root of {n} is about " + "{r}."
    if kind == "avg":
        xs = [rng.randint(3, 99) for _ in range(rng.choice([3, 4]))]
        p = rng.choice(
            [f"What is the average of {', '.join(map(str, xs))}?", f"Find the mean of {', '.join(map(str, xs))}."]
        )
        return p, f"({' + '.join(map(str, xs))}) / {len(xs)}", "The average is {r}."
    if kind == "chain":
        p = rng.choice(
            [
                f"Take {big}, subtract {a}, then multiply by {c}.",
                f"Divide {a * c} by {c}, then add {b}.",
                f"Add {a} and {b}, then multiply the sum by {c}.",
            ]
        )
        e = {0: f"({big} - {a}) * {c}", 1: f"{a * c} / {c} + {b}", 2: f"({a} + {b}) * {c}"}[
            [f"Take {big}, subtract {a}, then multiply by {c}.", f"Divide {a * c} by {c}, then add {b}.", f"Add {a} and {b}, then multiply the sum by {c}."].index(p)
        ]
        return p, e, "The result is {r}."
    if kind == "price":
        k = rng.randint(2, 40)
        unit = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12, 15])
        item = rng.choice(["pens", "tickets", "apples", "notebooks", "stamps", "cables"])
        p = rng.choice(
            [
                f"{k} {item} cost {unit} euros each. What is the total?",
                f"If one of these costs {unit} and I buy {k} {item}, what do I pay?",
            ]
        )
        return p, f"{k} * {unit}", "The total cost is {r}."
    if kind == "split":
        people = rng.randint(2, 9)
        each = rng.randint(5, 60)
        total = people * each
        p = rng.choice(
            [f"Split a bill of {total} between {people} people.", f"{total} euros shared equally by {people} friends is how much each?"]
        )
        return p, f"{total} / {people}", "Each person pays {r}."
    if kind == "area":
        w, h = rng.randint(2, 40), rng.randint(2, 40)
        p = rng.choice(
            [f"What is the area of a {w} by {h} rectangle?", f"A room is {w} metres by {h} metres. How many square metres is that?"]
        )
        return p, f"{w} * {h}", "The area is {r} square units."
    if kind == "rate":
        sp, hrs = rng.choice([40, 50, 60, 80, 90, 100, 120]), rng.choice([2, 3, 4, 5, 1.5, 2.5])
        p = rng.choice(
            [f"A car drives at {sp} km/h for {hrs} hours. How far does it go?", f"How far do you travel in {hrs} hours at {sp} kilometres per hour?"]
        )
        return p, f"{sp} * {hrs}", "The distance is {r} km."
    old, new = rng.choice([50, 80, 120, 200, 250]), rng.choice([60, 90, 150, 180, 300])
    p = rng.choice(
        [f"By what percent did a value change from {old} to {new}?", f"What is the percentage change from {old} to {new}?"]
    )
    return p, f"({new} - {old}) / {old} * 100", "The change is {r} percent."


def gen_calculator(b: Builder, target: int) -> None:
    attempts = 0
    while b.counts.get("calculator", 0) < target and attempts < target * 20:
        attempts += 1
        prompt, expr, template = _calc_spec(b.rng)
        out = b.run("calculator", {"expression": expr})
        res = result_of(out)
        if res is None:
            continue
        answer = template.format(r=res)
        b.accept(
            "calculator",
            [user(prompt), assistant_call(call("calculator", expression=expr)), tool(out), final(answer)],
            target,
        )


def gen_calculator_errors(b: Builder, target: int) -> None:
    attempts = 0
    while b.counts.get("calculator_error", 0) < target and attempts < target * 20:
        attempts += 1
        n = b.rng.randint(2, 900)
        prompt = b.rng.choice(
            [f"What is {n} divided by zero?", f"Compute {n} / 0.", f"Divide {n} by 0 for me.", f"How much is {n} over 0?"]
        )
        out = b.run("calculator", {"expression": f"{n} / 0"})
        b.accept(
            "calculator_error",
            [
                user(prompt),
                assistant_call(call("calculator", expression=f"{n} / 0")),
                tool(out),
                final(b.rng.choice([
                    "That is undefined: division by zero has no value.",
                    "The calculator reports division by zero, so there is no valid answer.",
                    "Division by zero is undefined, so the calculation cannot be completed.",
                ])),
            ],
            target,
        )


# ---------------------------------------------------------------- local_search
# (query, phrasings of the topic, fact sentence, words that must appear in the result)
SEARCH_TOPICS: List[Tuple[str, List[str], str, List[str]]] = [
    ("inverse document frequency", ["inverse document frequency", "IDF weighting", "why rare terms score higher"],
     "Inverse document frequency gives more weight to terms that occur in few documents.", ["inverse document frequency"]),
    ("term frequency k1", ["the k1 parameter", "term frequency saturation", "diminishing returns in BM25 scoring"],
     "More occurrences of a term raise the score, with diminishing returns controlled by the parameter k1.", ["k1"]),
    ("bm25 score negative", ["negative bm25 scores", "how FTS5 reports bm25", "the sign of the bm25 value"],
     "FTS5 returns a negative bm25 value, so better matches are more negative and queries order ascending.", ["negative"]),
    ("external content table", ["external content tables", "the recommended FTS5 schema", "how the FTS5 table avoids a second copy"],
     "The chunk text lives in an ordinary table and the FTS5 table indexes it without storing a second copy.", ["external content"]),
    ("quoting query terms", ["FTS5 query safety", "quoting search terms", "raw user text in MATCH"],
     "Production code quotes each term, because raw user text can raise a syntax error through stray quotes or operators.", ["quotes each term"]),
    ("porter stemming tokenizer", ["stemming", "the porter tokenizer", "how ranking and ranked match"],
     "The porter tokenizer maps words such as ranking and ranked to the same stem.", ["porter"]),
    ("inverted index posting list", ["the inverted index", "posting lists", "how a full-text index is structured"],
     "An inverted index stores, for every term, a posting list of document ids, term frequencies and positions.", ["posting list"]),
    ("why memory is separate", ["why memory is a separate store", "keeping memory outside the model", "benefits of an external memory"],
     "Weights are frozen after training, so notes live in a separate store that the user can inspect, that updates immediately, and where mistaken notes can be deleted.", ["frozen"]),
    ("memory unique key", ["the unique memory key", "what happens when a key is saved twice", "updating a note in place"],
     "Saving the same key again updates the note in place, so one key cannot accumulate contradictory values.", ["same key"]),
    ("memory writing policy", ["when memory should be written", "the memory writing policy", "what the assistant may save"],
     "The assistant should write memory only when the user asks, or when the user states and confirms a durable preference.", ["only when the user asks"]),
    ("memory retrieval policy", ["the memory retrieval policy", "how many notes to inject", "memory and context size"],
     "Retrieval should be conservative: take the top two or three notes and cap the injected text at a few hundred characters.", ["conservative"]),
    ("int8 checkpoint size", ["INT8 checkpoint size", "how much eight-bit weights save", "the size of a quantised model"],
     "A 49 million parameter model needs about 196 MB in float32, roughly 49 MB at eight bits and about 25 MB at four bits.", ["196 MB"]),
    ("int4 packing", ["INT4 packing", "four-bit weights", "how two values fit in one byte"],
     "INT4 values range from -8 to 7 and two signed four-bit values are packed into one byte.", ["packed into one byte"]),
    ("per-channel scale", ["per-channel scales", "the quantisation scale", "symmetric quantisation"],
     "Symmetric quantisation keeps one scale per output row, computed from the largest absolute weight in that row.", ["one scale per row"]),
    ("quantisation inference speed", ["whether quantisation is faster", "quantisation and CPU speed", "the speed effect of quantising"],
     "Weight-only quantisation shrinks the checkpoint but does not automatically make CPU inference faster.", ["does not automatically"]),
    ("measuring quantisation damage", ["checking quantisation error", "how to measure quantisation loss", "validation loss after quantising"],
     "Compare validation loss or perplexity before and after quantisation on the same batches, alongside the compression ratio.", ["before and after"]),
    ("embeddings stay in float", ["which layers stay in float", "the tied language-model head and quantisation", "why embeddings are not quantised"],
     "Embeddings and the tied head stay in float because quantising one would break the tie or duplicate the table.", ["tied"]),
    ("tool call format", ["the tool call format", "how a tool call is written", "the tool-calling marker"],
     "A tool call is a JSON object after a tool_call marker, with a name string and an arguments object.", ["tool_call"]),
    ("parsing malformed tool calls", ["parsing tool calls defensively", "handling invalid tool JSON", "tool call parse errors"],
     "The parser should extract the first balanced JSON object and return a readable error that the model can use to correct itself.", ["balanced"]),
    ("tool result shape", ["the shape of a tool result", "the tool result object", "fields in a tool result"],
     "A tool result has ok, content and a citations list, with an error string when ok is false.", ["citations"]),
    ("loop budgets", ["tool loop limits", "iteration budgets", "what limits the agent loop"],
     "Every loop needs a maximum number of iterations and tool calls per turn, a per-tool timeout and a context cap.", ["maximum number of iterations"]),
    ("calculator safety", ["how the calculator stays safe", "calculator allowlist", "why the calculator avoids eval"],
     "The calculator parses arithmetic with an allowlist instead of calling eval.", ["allowlist"]),
    ("rmsnorm", ["RMSNorm", "the normalisation layer", "how RMSNorm differs from LayerNorm"],
     "RMSNorm divides by the root mean square of the activations and multiplies by a learned weight, without subtracting the mean.", ["root mean square"]),
    ("grouped-query attention", ["grouped-query attention", "GQA", "how many key/value heads the model uses"],
     "Grouped-query attention uses fewer key/value heads than query heads, so with 8 query heads and 2 key/value heads the cache is a quarter of the multi-head size.", ["8 query heads"]),
    ("rotary position embeddings", ["RoPE", "rotary position embeddings", "how positions are encoded"],
     "RoPE rotates pairs of query and key dimensions by an angle proportional to position, so no learned position table is stored.", ["rotating pairs"]),
    ("swiglu feed-forward", ["the SwiGLU feed-forward network", "the intermediate size of the feed-forward layer", "SwiGLU"],
     "SwiGLU computes down(silu(gate(x)) * up(x)); TinyLM-50 uses an intermediate size of 1472 for hidden size 512.", ["1472"]),
    ("tied embeddings", ["tied embeddings", "weight tying", "sharing the input and output embedding"],
     "Tying the input embedding and output head saves vocabulary times hidden parameters, about 8.2 million here.", ["8.2 million"]),
    ("causal mask", ["the causal mask", "why attention is masked", "preventing attention to later positions"],
     "A causal mask stops a position from attending to later positions, so the model cannot copy the token it must predict.", ["causal mask"]),
    ("pre-norm", ["pre-norm placement", "where normalisation sits in a block", "why the model is pre-norm"],
     "Pre-norm applies normalisation to the input of each sub-layer, which keeps the identity path unscaled and is easier to train at depth.", ["pre-norm"]),
]

SEARCH_TEMPLATES = [
    "What does the library say about {t}?",
    "Look up {t} in the documents.",
    "Search the notes for {t}.",
    "Find what the documents say about {t}.",
    "Can you check the library for {t}?",
    "Is there anything in the library on {t}?",
    "Check the local documents about {t}.",
    "I need the library's explanation of {t}.",
    "Search for {t} and tell me what you find.",
    "What do my documents say about {t}?",
    "Use the library to explain {t}.",
    "Pull up the passage on {t}.",
]

UNKNOWN_TOPICS = [
    "zorblax dosage", "the migration of arctic terns", "the 1987 chess championship", "quantum gravity lattice",
    "the price of copper in 2019", "medieval pottery glazes", "the plot of a novel called Marlowe Hill",
    "gluten-free bread recipes", "marathon training plans", "the history of the Ottoman navy",
    "a framework called Frobnicate", "the zephyr protocol version nine",
]


def gen_search(b: Builder, target: int) -> None:
    attempts = 0
    while b.counts.get("local_search", 0) < target and attempts < target * 30:
        attempts += 1
        query, phrases, fact, must = b.rng.choice(SEARCH_TOPICS)
        phrase = b.rng.choice(phrases)
        prompt = b.rng.choice(SEARCH_TEMPLATES).format(t=phrase)
        limit = b.rng.choice([1, 1, 2, 3])
        out = b.run("local_search", {"query": query, "limit": limit})
        text = json.loads(out)["content"].lower()
        has_fact = all(term.lower() in text for term in must)
        msgs: List[Dict[str, Any]] = [user(prompt), assistant_call(call("local_search", query=query, limit=limit))]
        if not has_fact:
            continue
        # Full passages cost ~400 tokens (the tokenizer averages 1.75 chars/token on
        # JSON), so only a quarter of search examples keep the answer turn; the
        # rest still teach the call itself, which is what the evaluation scores.
        if len(out) <= MAX_RESULT_CHARS_WITH_FINAL and b.rng.random() < 0.25:
            msgs += [tool(out), final(f"{fact} [1]")]
        b.accept("local_search", msgs, target)


def gen_search_empty(b: Builder, target: int) -> None:
    attempts = 0
    while b.counts.get("local_search_empty", 0) < target and attempts < target * 30:
        attempts += 1
        topic = b.rng.choice(UNKNOWN_TOPICS)
        prompt = b.rng.choice(SEARCH_TEMPLATES).format(t=topic)
        out = b.run("local_search", {"query": topic, "limit": 3})
        if "No matching passages" not in out:
            continue
        b.accept(
            "local_search_empty",
            [
                user(prompt),
                assistant_call(call("local_search", query=topic, limit=3)),
                tool(out),
                final(b.rng.choice([
                    f"The local library has no passage about {topic}, so I cannot answer from it.",
                    f"I searched the local library for {topic} and found nothing, so I will not guess.",
                    f"No matching passage about {topic} exists in the library, so I cannot give a sourced answer.",
                ])),
            ],
            target,
        )


# --------------------------------------------------------------- read_document
DOC_SUMMARY = {
    "bm25_and_fts5.md": "It explains full-text search with BM25 ranking and SQLite FTS5: the inverted index, the external content table schema, safe query construction, stemming and chunking.",
    "memory_systems.md": "It explains persistent memory for a local assistant: why it is a separate store, a keyed-note schema, a conservative retrieval policy and a writing policy.",
    "quantization.md": "It covers weight-only quantisation: INT8 and INT4 per-channel schemes, what they do and do not buy, and how to measure the damage.",
    "tool_use_protocol.md": "It describes a text-based tool-calling protocol: the call format, defensive parsing, the result shape, and budgets and safety limits.",
    "transformer_architecture.md": "It describes the decoder-only transformer: pre-norm blocks, RMSNorm, grouped-query attention, RoPE, SwiGLU and tied embeddings.",
}
DOC_OPENERS = {
    "bm25_and_fts5.md": "full-text search needs an inverted index",
    "memory_systems.md": "memory is a separate store",
    "quantization.md": "quantising weights saves checkpoint size",
    "tool_use_protocol.md": "a textual protocol keeps tool calls simple",
    "transformer_architecture.md": "the model is a stack of pre-norm blocks",
}
READ_TEMPLATES = [
    "Read {f} and tell me what it covers.",
    "Please open the file {f}.",
    "What is {f} about? Read it.",
    "Read the first part of {f}.",
    "Give me a quick summary of {f}.",
    "Open {f} for me.",
    "Can you read {f} and describe it?",
    "Take a look inside {f}.",
    "Show what {f} says at the start.",
    "Summarise {f}.",
    "What does {f} contain?",
    "Pull the opening of {f}.",
    "Have a look at {f} and give me the gist.",
    "Read through {f} briefly.",
    "Tell me the main idea of {f}.",
    "Display the first lines of the file {f}.",
    "I would like an overview of {f}.",
    "Peek into {f} and report back.",
    "Load {f} and explain what it is for.",
    "Skim {f} for me.",
]


def gen_read(b: Builder, target: int) -> None:
    attempts = 0
    files = sorted(DOC_SUMMARY)
    while b.counts.get("read_document", 0) < target and attempts < target * 30:
        attempts += 1
        name = b.rng.choice(files)
        limit = b.rng.choice([300, 400, 500, 600])
        prompt = b.rng.choice(READ_TEMPLATES).format(f=name)
        out = b.run("read_document", {"path": name, "max_chars": limit})
        msgs = [user(prompt), assistant_call(call("read_document", path=name, max_chars=limit))]
        if len(out) <= MAX_RESULT_CHARS_WITH_FINAL:
            msgs += [tool(out), final(DOC_SUMMARY[name])]
        b.accept("read_document", msgs, target)


def gen_read_missing(b: Builder, target: int) -> None:
    missing = ["notes.txt", "report.md", "todo.md", "readme.md", "budget.pdf", "meeting_notes.md", "draft.txt", "summary.md"]
    attempts = 0
    while b.counts.get("read_missing", 0) < target and attempts < target * 30:
        attempts += 1
        name = b.rng.choice(missing)
        prompt = b.rng.choice(["Read {f}.", "Open {f} and summarise it.", "What is in {f}?", "Show me {f}."]).format(f=name)
        out = b.run("read_document", {"path": name, "max_chars": 500})
        b.accept(
            "read_missing",
            [
                user(prompt),
                assistant_call(call("read_document", path=name, max_chars=500)),
                tool(out),
                final(f"I could not find {name} in the document library. I can list the available files if that helps."),
            ],
            target,
        )


# ------------------------------------------------------------------- list_files
def gen_list(b: Builder, target: int) -> None:
    variants = [
        ("*.md", ["Which Markdown files do you have?", "Show every .md file.", "What markdown documents exist in the library?", "List all the Markdown notes.", "Which .md documents can you open?", "Give me the names of the Markdown files.", "How many Markdown documents are there?", "Tell me which markdown notes exist."]),
        ("*", ["What files are in the library?", "List everything you can read.", "Show me all the documents.", "What documents do you have access to?", "What can I ask you to read?", "Show the whole library contents.", "Which documents are indexed?", "Name every file you can open."]),
        ("*.pdf", ["Do you have any PDFs?", "Are there PDF documents available?", "Show me the PDF files.", "Any .pdf files around?", "Is there a PDF I can ask about?"]),
        ("*.txt", ["Are there any text files?", "List the .txt files.", "Do you have plain text documents?", "Any .txt notes available?", "Show me files ending in .txt."]),
        ("*.py", ["Are there any Python files in the library?", "List the .py files.", "Do you have any source code files?", "Show the Python scripts available."]),
        ("quant*", ["Is there a document whose name starts with quant?", "List files starting with quant.", "Find documents named quant something."]),
        ("*memory*", ["Which documents have memory in the name?", "Find files with memory in their name.", "Is there a file about memory by name?"]),
    ]
    attempts = 0
    while b.counts.get("list_files", 0) < target and attempts < target * 30:
        attempts += 1
        pattern, prompts = b.rng.choice(variants)
        prompt = b.rng.choice(prompts)
        limit = b.rng.choice([10, 20, 50])
        out = b.run("list_files", {"pattern": pattern, "max_results": limit})
        content = json.loads(out)["content"]
        if content.startswith("No matching"):
            answer = b.rng.choice([
                f"No documents match {pattern} in the library.",
                f"The library has no files matching {pattern}.",
            ])
        else:
            names = re.findall(r"^(\S+) \(", content, flags=re.M)
            answer = f"The library has {len(names)} matching document" + ("s" if len(names) != 1 else "") + ": " + ", ".join(names) + "."
        b.accept(
            "list_files",
            [user(prompt), assistant_call(call("list_files", pattern=pattern, max_results=limit)), tool(out), final(answer)],
            target,
        )


# --------------------------------------------------------------------- memory
SLOTS = {
    "timezone": ("timezone", ["Europe/Berlin", "America/Chicago", "Asia/Tokyo", "Europe/London", "Australia/Sydney", "America/Denver"], "preference,time"),
    "editor": ("editor", ["Vim", "VS Code", "Emacs", "Neovim", "Sublime Text", "Helix"], "preference,tools"),
    "language": ("favourite language", ["Python", "Rust", "Go", "TypeScript", "Kotlin", "OCaml"], "preference,coding"),
    "city": ("city", ["Lisbon", "Toronto", "Nairobi", "Osaka", "Prague", "Lima"], "profile"),
    "project_name": ("project name", ["Atlas", "Nimbus", "Orion", "Helios", "Kestrel", "Quartz"], "project"),
    "deadline": ("deadline", ["March 3", "June 18", "the end of May", "next Friday", "October 9", "January 12"], "project,time"),
    "theme": ("preferred theme", ["dark", "light", "solarized", "high contrast"], "preference,ui"),
    "pet": ("dog's name", ["Biscuit", "Pepper", "Juno", "Waffles", "Milo", "Nova"], "profile"),
    "colour": ("favourite colour", ["green", "orange", "teal", "purple", "red", "yellow"], "preference"),
    "database": ("database", ["PostgreSQL", "SQLite", "MySQL", "DuckDB"], "project,tools"),
    "team": ("team name", ["Platform", "Growth", "Payments", "Research", "Mobile"], "profile,work"),
    "units": ("preferred units", ["imperial", "metric"], "preference"),
    "meeting_day": ("weekly meeting day", ["Monday", "Tuesday", "Wednesday", "Thursday"], "schedule"),
    "package_manager": ("package manager", ["pnpm", "npm", "yarn", "pip", "uv", "cargo"], "preference,tools"),
}
SAVE_TEMPLATES = [
    "Remember that my {s} is {v}.",
    "Please save a note: my {s} is {v}.",
    "Keep in mind that my {s} is {v}.",
    "Note down that my {s} is {v}.",
    "Don't forget my {s} is {v}.",
    "Store this for later: my {s} is {v}.",
    "From now on, my {s} is {v}. Please remember it.",
    "I want you to remember my {s}: {v}.",
]
RETRIEVE_TEMPLATES = [
    "What is my {s}?",
    "Do you remember my {s}?",
    "What {s} did I tell you?",
    "Remind me what my {s} is.",
    "Which {s} did I ask you to save?",
    "Can you tell me my {s} from your notes?",
    "Look up my saved {s}.",
    "What did I say my {s} was?",
]


def _slot(rng: random.Random) -> Tuple[str, str, str, str]:
    key, (phrase, values, tags) = rng.choice(list(SLOTS.items()))
    return key, phrase, rng.choice(values), tags


def gen_save(b: Builder, target: int) -> None:
    attempts = 0
    while b.counts.get("save_memory", 0) < target and attempts < target * 30:
        attempts += 1
        key, phrase, value, tags = _slot(b.rng)
        prompt = b.rng.choice(SAVE_TEMPLATES).format(s=phrase, v=value)
        importance = b.rng.choice([0.5, 0.6, 0.7, 0.8, 0.9])
        mem_key = f"user_{key}"
        content = f"The user's {phrase} is {value}."
        store = MemoryStore(db_path=str(Path(b.tmp) / f"m{attempts}.db"))
        out = b.fresh_memory_run(
            "save_memory", {"key": mem_key, "content": content, "tags": tags, "importance": importance}, store
        )
        b.accept(
            "save_memory",
            [
                user(prompt),
                assistant_call(call("save_memory", key=mem_key, content=content, tags=tags, importance=importance)),
                tool(out),
                final(b.rng.choice([
                    f"Saved. I will remember that your {phrase} is {value}.",
                    f"Got it. I stored your {phrase} ({value}) under the key {mem_key}.",
                    f"Done. Your {phrase} is saved as {value}.",
                ])),
            ],
            target,
        )


def gen_retrieve(b: Builder, target: int) -> None:
    attempts = 0
    while b.counts.get("retrieve_memory", 0) < target and attempts < target * 30:
        attempts += 1
        key, phrase, value, tags = _slot(b.rng)
        prompt = b.rng.choice(RETRIEVE_TEMPLATES).format(s=phrase)
        query = phrase.replace("favourite ", "").replace("preferred ", "").replace("weekly ", "").replace("'s", "")
        store = MemoryStore(db_path=str(Path(b.tmp) / f"r{attempts}.db"))
        found = b.rng.random() < 0.85
        if found:
            store.save(f"user_{key}", f"The user's {phrase} is {value}.", tags=tags, importance=0.7)
        out = b.fresh_memory_run("retrieve_memory", {"query": query, "limit": 3}, store)
        if found and "No matching" in out:
            continue
        if found:
            answer = b.rng.choice([
                f"Your {phrase} is {value}.",
                f"You told me your {phrase} is {value}.",
                f"According to my notes, your {phrase} is {value}.",
            ])
        else:
            answer = b.rng.choice([
                f"I have no saved note about your {phrase}. You can tell me and I will remember it.",
                f"I could not find a {phrase} in memory, so I do not know it yet.",
            ])
        b.accept(
            "retrieve_memory",
            [user(prompt), assistant_call(call("retrieve_memory", query=query, limit=3)), tool(out), final(answer)],
            target,
        )


# -------------------------------------------------------------------- no tool
DEFINITIONS = {
    "a variable": "A variable is a named place in a program that holds a value which can change.",
    "a function": "A function is a reusable block of code that takes inputs, does some work and can return a result.",
    "a loop": "A loop repeats a block of code until a condition stops it.",
    "an array": "An array is an ordered collection of items that you reach by position.",
    "a hash map": "A hash map stores key-value pairs and finds a value quickly by hashing its key.",
    "an API": "An API is a defined way for one program to ask another program to do something.",
    "a compiler": "A compiler translates source code written by a person into machine code that a computer can run.",
    "a database": "A database is an organised store of data that can be queried and updated.",
    "an algorithm": "An algorithm is a finite list of steps for solving a problem.",
    "recursion": "Recursion is when a function solves a problem by calling itself on a smaller version of it.",
    "a bug": "A bug is a mistake in a program that makes it behave in a way its author did not intend.",
    "a git commit": "A git commit is a saved snapshot of a project's files with a message describing the change.",
    "encryption": "Encryption scrambles data with a key so that only someone with the right key can read it.",
    "an IP address": "An IP address is a number that identifies a device on a network.",
    "a URL": "A URL is the address of a resource on the web.",
    "an operating system": "An operating system manages a computer's hardware and runs the programs on it.",
    "a byte": "A byte is a unit of data made of eight bits.",
    "RAM": "RAM is the fast, temporary memory a computer uses for the data it is working on right now.",
    "a CPU": "A CPU is the processor that carries out a computer's instructions.",
    "a queue": "A queue is a collection where the first item added is the first item removed.",
    "a stack": "A stack is a collection where the last item added is the first item removed.",
    "a token": "A token is a small piece of text, such as a word or part of a word, that a language model reads as one unit.",
    "an embedding": "An embedding is a list of numbers that represents a token or piece of text so a model can compare meanings.",
    "overfitting": "Overfitting is when a model memorises its training data and then performs poorly on new data.",
    "a cache": "A cache is a small, fast store that keeps recent or frequent data close at hand so it can be reached quickly.",
    "a thread": "A thread is a sequence of instructions that a program can run alongside other threads.",
    "a server": "A server is a computer or program that answers requests from other computers or programs.",
    "a browser": "A browser is a program for viewing web pages.",
    "a firewall": "A firewall is a filter that allows or blocks network traffic according to rules.",
    "a pixel": "A pixel is the smallest dot of colour in a digital image.",
    "an integer": "An integer is a whole number with no fractional part.",
    "a string": "A string is a sequence of characters, such as a word or sentence.",
    "a boolean": "A boolean is a value that is either true or false.",
    "a library": "A library is a collection of ready-made code that other programs can reuse.",
    "a gradient": "A gradient says how much and in which direction a value changes, and training uses it to adjust a model.",
    "a loss function": "A loss function measures how wrong a model's predictions are, and training tries to make it smaller.",
    "a neural network": "A neural network is a model made of layers of simple units whose connections are learned from data.",
    "a dataset": "A dataset is a collection of examples used to train or test a model.",
    "a parameter": "A parameter is a number inside a model that is adjusted during training.",
}
DEF_TEMPLATES = [
    "What is {t}?", "Explain {t} briefly.", "Define {t} in plain words.", "Can you tell me what {t} is?",
    "Give me a short definition of {t}.", "What does {t} mean?", "Describe {t} in a sentence.", "I keep hearing about {t}. What is it?",
]
SMALLTALK = [
    ("Hello!", "Hello! I am TinyLM-50, a small offline assistant. I can calculate, search the local document library and remember notes for you."),
    ("Hi there", "Hi! How can I help? I can calculate, search the library and keep notes."),
    ("Good morning", "Good morning! What would you like to do today?"),
    ("Good evening", "Good evening! How can I help?"),
    ("Hey", "Hey! What can I do for you?"),
    ("Thanks!", "You are welcome."),
    ("Thank you so much", "Happy to help."),
    ("That was helpful, thanks", "Glad it helped."),
    ("Goodbye", "Goodbye! Come back any time."),
    ("Bye for now", "See you later."),
    ("See you tomorrow", "See you tomorrow!"),
    ("Who are you?", "I am TinyLM-50, a small offline assistant that uses tools such as a calculator and a local document search."),
    ("What is your name?", "My name is TinyLM-50."),
    ("What can you do?", f"I can do exact arithmetic, search and read the local library, and save and recall notes. My tools are {TOOL_NAMES}."),
    ("Which tools do you have?", f"My tools are {TOOL_NAMES}."),
    ("Are you online?", "No. I run fully offline and only use the local tools and documents."),
    ("Can you browse the internet?", "No. I am offline and can only use my local tools."),
    ("How are you?", "I am working fine, thank you. How can I help?"),
    ("Write a short greeting for a new teammate.", "Welcome to the team! We are glad you are here."),
    ("Say hello in French.", "Bonjour."),
    ("How do you say thank you in Spanish?", "Gracias."),
    ("How do you say good night in German?", "Gute Nacht."),
    ("Give me a motivational one-liner.", "Small steps every day add up to big results."),
    ("Write a friendly sign-off for an email.", "Best wishes, and thank you for your time."),
    ("Wish me luck on my exam.", "Good luck on your exam. You have got this!"),
    ("Congratulate me on finishing my project.", "Congratulations on finishing your project!"),
]


def gen_no_tool(b: Builder, target: int) -> None:
    items: List[Tuple[str, str]] = list(SMALLTALK)
    for term, answer in DEFINITIONS.items():
        for template in b.rng.sample(DEF_TEMPLATES, 4):
            items.append((template.format(t=term), answer))
    b.rng.shuffle(items)
    for prompt, answer in items:
        b.accept("no_tool", [user(prompt), final(answer)], target)


UNAVAILABLE = [
    ("email", ["Email {p} about the meeting.", "Write and send a message to {p} by email.", "Please mail {p} the report.", "Draft an email to {p} and send it.", "Reply to the last email from {p}."], "send emails"),
    ("calendar", ["Book a meeting with {p} for tomorrow.", "Add a dentist appointment to my calendar.", "Schedule a call with {p} on Friday.", "Put a lunch with {p} on my calendar.", "Move my Monday meeting to Tuesday."], "manage a calendar"),
    ("weather", ["What is the weather in Paris today?", "Will it rain tomorrow in Chicago?", "Tell me the current temperature in Tokyo.", "Is it sunny in Madrid right now?", "What is the forecast for this weekend?"], "look up the weather"),
    ("web", ["Search the web for the latest news.", "Open google.com and find a pizza recipe.", "Browse to a website and read its headlines.", "Look online for the cheapest flights.", "Download the latest release from GitHub."], "browse the web"),
    ("phone", ["Call {p} for me.", "Send a text message to {p}.", "Ring my mother.", "Text {p} that I am running late.", "Dial the office."], "make calls or send texts"),
    ("alarm", ["Set an alarm for 7 am.", "Remind me in ten minutes to stretch.", "Start a timer for five minutes.", "Wake me up at six tomorrow.", "Remind me to call {p} at noon."], "set alarms or timers"),
    ("music", ["Play some jazz.", "Skip to the next song.", "Turn the volume up.", "Put on a relaxing playlist.", "Pause the music."], "play music"),
    ("shell", ["Run ls in a terminal.", "Execute this shell command for me.", "Delete my temporary files with a script.", "Install a package with pip for me.", "Restart the server from the command line."], "run shell commands"),
    ("order", ["Order a pizza for dinner.", "Buy a train ticket to Berlin.", "Reserve a table for two tonight.", "Book me a hotel in Rome.", "Order groceries for the week."], "place orders or bookings"),
]
PEOPLE = ["my manager", "Sam", "the team", "my landlord", "Priya", "my sister", "the client", "Dr. Lee"]


def gen_unavailable(b: Builder, target: int) -> None:
    attempts = 0
    while b.counts.get("unavailable_tool", 0) < target and attempts < target * 30:
        attempts += 1
        name, prompts, verb = b.rng.choice(UNAVAILABLE)
        prompt = b.rng.choice(prompts).format(p=b.rng.choice(PEOPLE))
        answer = b.rng.choice([
            f"I do not have a {name} tool, so I cannot {verb}. My available tools are {TOOL_NAMES}.",
            f"Sorry, I cannot {verb}. I only have {TOOL_NAMES}.",
            f"That needs a {name} tool, which I do not have. I can use {TOOL_NAMES}.",
        ])
        b.accept("unavailable_tool", [user(prompt), final(answer)], target)


# ------------------------------------------------------------------ multi-tool
def gen_multi(b: Builder, target: int) -> None:
    attempts = 0
    while b.counts.get("multi_tool", 0) < target and attempts < target * 30:
        attempts += 1
        mode = b.rng.choice(["calc_calc", "calc_calc", "search_calc", "save_calc", "calc_search"])
        a1, a2, a3, a4 = (b.rng.randint(3, 60) for _ in range(4))
        if mode == "calc_calc":
            e1, e2 = f"{a1} * {a2}", f"{a3} + {a4}"
            prompt = b.rng.choice([
                f"Compute {a1} times {a2} and also {a3} plus {a4}.",
                f"Give me both {a1} * {a2} and {a3} + {a4}.",
                f"What are {a1} x {a2} and {a3} + {a4}?",
            ])
            o1, o2 = b.run("calculator", {"expression": e1}), b.run("calculator", {"expression": e2})
            r1, r2 = result_of(o1), result_of(o2)
            msgs = [
                user(prompt),
                assistant_call(call("calculator", expression=e1), call("calculator", expression=e2)),
                tool(o1), tool(o2),
                final(f"{e1} = {r1} and {e2} = {r2}."),
            ]
        elif mode in ("search_calc", "calc_search"):
            query, phrases, _fact, _must = b.rng.choice(SEARCH_TOPICS)
            phrase = b.rng.choice(phrases)
            e = f"{a1} * {a2}"
            if mode == "search_calc":
                prompt = b.rng.choice([
                    f"Search the library for {phrase} and also calculate {a1} times {a2}.",
                    f"Find {phrase} in the documents, and compute {a1} * {a2}.",
                    f"Check the library about {phrase}, plus work out {a1} x {a2}.",
                ])
                calls = (call("local_search", query=query, limit=1), call("calculator", expression=e))
            else:
                prompt = b.rng.choice([
                    f"Calculate {a1} times {a2} and also look up {phrase} in the library.",
                    f"Work out {a1} * {a2}, then search the documents for {phrase}.",
                ])
                calls = (call("calculator", expression=e), call("local_search", query=query, limit=1))
            # Call-only: the search passage alone is too long to keep with an answer.
            msgs = [user(prompt), assistant_call(*calls)]
        else:
            key, phrase, value, tags = _slot(b.rng)
            e = f"{a1} + {a2}"
            prompt = b.rng.choice([
                f"Remember that my {phrase} is {value}, and also compute {a1} plus {a2}.",
                f"Save my {phrase} as {value} and work out {a1} + {a2}.",
            ])
            mem_key, content = f"user_{key}", f"The user's {phrase} is {value}."
            store = MemoryStore(db_path=str(Path(b.tmp) / f"mm{attempts}.db"))
            o1 = b.fresh_memory_run("save_memory", {"key": mem_key, "content": content, "tags": tags, "importance": 0.7}, store)
            o2 = b.run("calculator", {"expression": e})
            msgs = [
                user(prompt),
                assistant_call(call("save_memory", key=mem_key, content=content, tags=tags, importance=0.7), call("calculator", expression=e)),
                tool(o1), tool(o2),
                final(f"Saved your {phrase} as {value}. Also, {e} = {result_of(o2)}."),
            ]
        b.accept("multi_tool", msgs, target)


# ------------------------------------------------------------------------ main
DEFAULT_TARGETS = {
    "calculator": 340, "calculator_error": 30, "local_search": 240, "local_search_empty": 40,
    "read_document": 100, "read_missing": 24, "list_files": 80, "save_memory": 150,
    "retrieve_memory": 120, "no_tool": 190, "unavailable_tool": 90, "multi_tool": 90,
}


def generate(seed: int = 1234, scale: float = 1.0, include_hand_written: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    b = Builder(seed)
    t = {k: max(1, int(v * scale)) for k, v in DEFAULT_TARGETS.items()}
    gen_calculator(b, t["calculator"])
    gen_calculator_errors(b, t["calculator_error"])
    gen_search(b, t["local_search"])
    gen_search_empty(b, t["local_search_empty"])
    gen_read(b, t["read_document"])
    gen_read_missing(b, t["read_missing"])
    gen_list(b, t["list_files"])
    gen_save(b, t["save_memory"])
    gen_retrieve(b, t["retrieve_memory"])
    gen_no_tool(b, t["no_tool"])
    gen_unavailable(b, t["unavailable_tool"])
    gen_multi(b, t["multi_tool"])
    records = list(b.out)
    if include_hand_written and HAND_WRITTEN.exists():
        for line in HAND_WRITTEN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            prompts = [_norm(m["content"]) for m in sample["messages"] if m["role"] == "user"]
            if all(p not in b.seen_prompts for p in prompts):
                records.append(sample)
                b.counts["hand_written"] = b.counts.get("hand_written", 0) + 1
    random.Random(seed).shuffle(records)
    return records, b.counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the TinyLM-50 tool-use SFT set")
    parser.add_argument("--out", default="datasets/instructions/tiny50_sft_v2.jsonl")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every category target")
    parser.add_argument("--no-hand-written", action="store_true", help="omit the 20 curated conversations")
    args = parser.parse_args(argv)
    records, counts = generate(args.seed, args.scale, include_hand_written=not args.no_hand_written)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} conversations to {out}")
    for category, count in sorted(counts.items()):
        print(f"  {category:<20} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
