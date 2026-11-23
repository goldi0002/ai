"""Build a fully wired agent (tools + research index + memory) from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from memory.store import MemoryStore
from model.tokenizer import TinyTokenizer
from tools.base import Tool
from tools.calculator import Calculator
from tools.files import ListFiles, ReadDocument
from tools.local_search import LocalSearch
from tools.memory_tools import RetrieveMemory, SaveMemory
from tools.registry import ToolRegistry
from tools.research import ResearchEngine

from .agent import Agent, AgentConfig


def build_registry(
    db_path: str | Path = "storage/tinylm.db",
    document_root: str | Path = "datasets/documents",
    memory: Optional[MemoryStore] = None,
) -> Tuple[ToolRegistry, ResearchEngine, MemoryStore]:
    engine = ResearchEngine(db_path=db_path, root=document_root)
    memory = memory or MemoryStore(db_path=db_path)
    tools = [
        Calculator(),
        LocalSearch(engine),
        ReadDocument(document_root),
        ListFiles(document_root),
        RetrieveMemory(memory),
        SaveMemory(memory),
    ]
    return ToolRegistry(tools), engine, memory


def build_agent(
    model,
    tokenizer: TinyTokenizer,
    settings: Dict[str, Any],
    memory: Optional[MemoryStore] = None,
) -> Agent:
    registry, _engine, memory = build_registry(
        db_path=settings.get("db_path", "storage/tinylm.db"),
        document_root=settings.get("document_root", "datasets/documents"),
        memory=memory,
    )
    config = AgentConfig(
        max_iterations=int(settings.get("max_iterations", 4)),
        max_new_tokens=int(settings.get("max_new_tokens", 256)),
        max_context_tokens=int(settings.get("max_context_tokens", 1024)),
        temperature=float(settings.get("temperature", 0.7)),
        top_p=float(settings.get("top_p", 0.9)),
        tool_timeout=float(settings.get("tool_timeout_seconds", 5.0)),
    )
    return Agent(
        model,
        tokenizer,
        registry,
        memory=memory,
        config=config,
        system_prompt=settings.get("system_prompt"),
    )
