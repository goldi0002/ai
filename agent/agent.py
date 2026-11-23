"""Tool-using agent loop.

The loop is explicit: build a prompt, generate, parse tool calls, execute
them, append the observations, repeat until the model answers or the
iteration budget is exhausted. Tool failures are fed back to the model instead
of aborting the conversation.

For tests (and for scripted demos) a ``generate_fn(prompt_ids, max_new_tokens)``
can replace the model; the rest of the loop is identical.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from memory.store import MemoryStore
from model.tokenizer import TinyTokenizer
from tools.base import ToolResult
from tools.registry import ToolRegistry

from .parser import ParsedOutput, ToolCall, parse_tool_calls, strip_special_tokens
from .prompts import COMPACT_SYSTEM_PROMPT, build_system_prompt, render_prompt

GenerateFn = Callable[[List[int], int], List[int]]


@dataclass
class AgentConfig:
    max_iterations: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    tool_timeout: float = 5.0
    max_context_tokens: int = 1024
    max_tool_calls_per_turn: int = 3
    device: str = "auto"


@dataclass
class AgentResponse:
    text: str
    iterations: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    tokens_generated: int = 0
    prompt_tokens: int = 0
    stopped_reason: str = "final"
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Agent:
    def __init__(
        self,
        model: Optional[torch.nn.Module],
        tokenizer: Optional[TinyTokenizer],
        registry: ToolRegistry,
        memory: Optional[MemoryStore] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
        generate_fn: Optional[GenerateFn] = None,
    ) -> None:
        if model is None and generate_fn is None:
            raise ValueError("Agent needs a model or a generate_fn")
        if tokenizer is None:
            raise ValueError("Agent needs a tokenizer")
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.memory = memory
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt or COMPACT_SYSTEM_PROMPT
        self.generate_fn = generate_fn
        self.history: List[Tuple[str, str]] = []
        if model is not None and self.config.device != "auto":
            self.model.to(torch.device(self.config.device))
        self._device = (
            next(model.parameters()).device
            if model is not None
            else torch.device("cpu")
        )
        # The prompt must leave room for the reply inside the model's RoPE range.
        self._model_max_seq = int(model.config.max_seq_len) if model is not None else 4096

    def _prompt_budget(self) -> int:
        new_tokens = max(8, self.config.max_new_tokens)
        return max(32, min(self.config.max_context_tokens, self._model_max_seq - new_tokens))

    # ------------------------------------------------------------------ model
    def _generate(self, prompt_ids: List[int], max_new_tokens: int) -> List[int]:
        if self.generate_fn is not None:
            return [int(token) for token in self.generate_fn(prompt_ids, max_new_tokens)]
        assert self.model is not None
        max_new_tokens = max(1, min(max_new_tokens, self._model_max_seq - len(prompt_ids)))
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                stop_token_ids={self.tokenizer.id("<eos>"), self.tokenizer.id("<user>")},
            )
        if was_training:
            self.model.train()
        return output[0, len(prompt_ids) :].tolist()

    # ------------------------------------------------------------------ loop
    def chat(
        self,
        message: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> AgentResponse:
        started = time.time()
        self.history = list(history or []) + [("user", message)]
        memory_context = ""
        if self.memory is not None:
            try:
                memory_context = self.memory.to_context(message, limit=3)
            except Exception:  # memory is best-effort context, never fatal
                memory_context = ""
        system_prompt = self.system_prompt or build_system_prompt(self.registry, memory_context)

        response = AgentResponse(text="", iterations=0)
        seen_calls: Dict[str, str] = {}
        last_text = ""

        for iteration in range(1, self.config.max_iterations + 1):
            response.iterations = iteration
            prompt = render_prompt(
                self.tokenizer,
                system_prompt,
                self.history,
                max_tokens=self._prompt_budget(),
            )
            prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
            response.prompt_tokens = len(prompt_ids)
            new_ids = self._generate(prompt_ids, self.config.max_new_tokens)
            raw = self.tokenizer.decode(new_ids, skip_special_tokens=False)
            response.tokens_generated += len(new_ids)
            parsed: ParsedOutput = parse_tool_calls(raw)
            last_text = parsed.text or last_text
            for problem in parsed.errors:
                response.errors.append(problem)

            if not parsed.tool_calls:
                if parsed.errors and iteration < self.config.max_iterations:
                    # Feed the parse error back once so the model can repair
                    # truncated or malformed JSON instead of returning it.
                    self.history.append(("assistant", raw.strip()))
                    self.history.append(
                        (
                            "tool",
                            json.dumps({"ok": False, "error": parsed.errors[0]}),
                        )
                    )
                    continue
                text = parsed.text.strip()
                if not text:
                    text = (
                        "The model produced no final answer within the token budget. "
                        "Try a larger --max-new-tokens or a different prompt."
                    )
                response.text = text
                response.stopped_reason = "final"
                break

            # Keep the raw assistant turn (including its calls) in history, then
            # append one observation per executed call.
            self.history.append(("assistant", raw.strip()))
            calls = parsed.tool_calls[: self.config.max_tool_calls_per_turn]
            if len(parsed.tool_calls) > len(calls):
                response.errors.append(
                    f"only {len(calls)} of {len(parsed.tool_calls)} tool calls executed this turn"
                )

            for call in calls:
                signature = call.name + ":" + json.dumps(call.arguments, sort_keys=True)
                if signature in seen_calls:
                    observation = json.dumps(
                        {
                            "ok": False,
                            "error": "this exact tool call was already executed; "
                            "use the earlier result or answer now",
                        }
                    )
                    self.history.append(("tool", observation))
                    response.errors.append(f"repeated tool call: {call.name}")
                    response.stopped_reason = "repeated_tool_call"
                    break
                result = self.registry.execute(
                    call.name, call.arguments, timeout=self.config.tool_timeout
                )
                seen_calls[signature] = result.content
                observation = result.to_json()
                self.history.append(("tool", observation))
                response.tool_calls.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "ok": result.ok,
                        "content": result.content,
                        "error": result.error,
                    }
                )
                for citation in result.citations:
                    if citation not in response.citations:
                        response.citations.append(citation)
            if response.stopped_reason == "repeated_tool_call":
                response.text = last_text or (
                    "I stopped because the model repeated the same tool call instead of "
                    "using the result."
                )
                break
        else:
            response.text = last_text or (
                f"No final answer after {self.config.max_iterations} iterations."
            )
            response.stopped_reason = "max_iterations"

        response.elapsed_sec = time.time() - started
        return response

    # ------------------------------------------------------------------ misc
    def reset(self) -> None:
        self.history = []

    def run_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a single tool directly (used by the CLI and tests)."""
        return self.registry.execute(name, arguments, timeout=self.config.tool_timeout)
