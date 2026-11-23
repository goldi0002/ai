"""Prompt construction for the TinyLM-50 tool-using agent."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from model.tokenizer import TinyTokenizer
from tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are TinyLM-50, a small offline assistant that answers with the help of tools.

Rules:
1. Use calculator for every arithmetic result you report. Do not do mental arithmetic.
2. Use local_search when the answer may be in the local document library. Cite the passages you used as [1], [2], matching the order the tool returned them.
3. Never invent a citation, quotation or file path. If local_search returns nothing, say that the local library has no matching passage.
4. Use retrieve_memory before answering questions about the user's saved preferences. Use save_memory when the user asks you to remember something.
5. If no tool is needed, answer directly.

Tool call format (exactly one JSON object per call, on its own line):
<tool_call>{"name": "calculator", "arguments": {"expression": "17*23+4"}}

After a tool call, stop and wait. The result arrives as a <tool_result> message.
When you are ready to answer, write the answer after <final>."""

FINAL_MARKER = "<final>"

# Matches the system prompt used in datasets/instructions/tiny50_sft.jsonl. A
# 256-token model cannot afford the long prompt above, and deviating from the
# fine-tuning distribution visibly degrades tool-call formatting.
COMPACT_SYSTEM_PROMPT = (
    "You are TinyLM-50, a small offline assistant that answers with the help of tools."
)


def build_system_prompt(
    registry: ToolRegistry,
    memory_context: str = "",
    extra: str = "",
) -> str:
    parts = [SYSTEM_PROMPT, "", "Available tools:", registry.describe()]
    if memory_context:
        parts += ["", "Relevant memory (saved notes, may be stale):", memory_context]
    if extra:
        parts += ["", extra]
    return "\n".join(parts).strip()


def render_prompt(
    tokenizer: TinyTokenizer,
    system_prompt: str,
    turns: Sequence[Tuple[str, str]],
    max_tokens: int = 1024,
) -> str:
    """Render system + conversation turns into the training token format.

    Turns are ``("user"|"assistant"|"tool", text)``. The prompt is trimmed from
    the oldest turn until it fits ``max_tokens`` (special tokens included).
    """
    role_tokens = {
        "user": "<user>",
        "assistant": "<assistant>",
        "tool": "<tool_result>",
    }
    for role, _ in turns:
        if role not in role_tokens:
            raise ValueError(f"unsupported turn role {role!r}")

    def render(selected: Iterable[Tuple[str, str]]) -> str:
        # No separators between turns: this is exactly the layout produced by
        # training.data.render_conversation.
        pieces = [f"<system>{system_prompt}"]
        for role, text in selected:
            pieces.append(f"<{role}>{text}")
        # The model continues an assistant turn.
        pieces.append("<assistant>")
        return "".join(pieces)

    selected: List[Tuple[str, str]] = list(turns)
    prompt = render(selected)
    while len(tokenizer.encode(prompt)) > max_tokens and len(selected) > 1:
        selected.pop(0)
        prompt = render(selected)
    if len(tokenizer.encode(prompt)) > max_tokens:
        # A single oversized turn: keep its tail, which is where the question is.
        role, text = selected[0]
        encoded = tokenizer.encode(text)
        keep = max(16, max_tokens // 2)
        selected = [(role, tokenizer.decode(encoded[-keep:], skip_special_tokens=True))]
        prompt = render(selected)
    # The fixed system prompt itself may be too long: keep its head (rules and
    # tool list) and trim from the end until the whole prompt fits.
    while len(tokenizer.encode(prompt)) > max_tokens:
        system_ids = tokenizer.encode(system_prompt)
        if len(system_ids) <= 16:
            break
        excess = len(tokenizer.encode(prompt)) - max_tokens + 8
        system_prompt = tokenizer.decode(
            system_ids[: max(8, len(system_ids) - excess)], skip_special_tokens=True
        )
        prompt = render(selected)
    return prompt
