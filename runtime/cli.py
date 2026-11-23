"""Command-line interface for TinyLM-50.

    python -m runtime.cli info
    python -m runtime.cli generate "The capital of France is"
    python -m runtime.cli chat
    python -m runtime.cli research ingest --dir datasets/documents
    python -m runtime.cli research search "BM25"
    python -m runtime.cli memory save user_timezone "Europe/Berlin"
    python -m runtime.cli tools
    python -m runtime.cli benchmark --out storage/benchmark.json
    python -m runtime.cli quantize --bits 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from memory.store import MemoryStore
from model.config import TinyLMConfig
from training.common import load_yaml, section
from tools.research import ResearchEngine

CONFIG_PATH = Path("configs/tiny50.yaml")


def _config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        return load_yaml(CONFIG_PATH)
    return {}


def _settings() -> Dict[str, Any]:
    raw = _config()
    runtime = dict(section(raw, "runtime"))
    agent = dict(section(raw, "agent"))
    runtime.setdefault("checkpoint", "checkpoints/latest.pt")
    runtime.setdefault("tokenizer", "data/tokenizer/tiny50_sp.model")
    runtime.update({key: value for key, value in agent.items() if key not in runtime})
    return runtime


def _require_checkpoint(path: str) -> Optional[str]:
    if not Path(path).exists():
        print(
            f"error: no checkpoint at {path}.\n"
            "Train one first, e.g. `python -m training.train --max-steps 50`.",
            file=sys.stderr,
        )
        return None
    return path


# ------------------------------------------------------------------- commands
def cmd_info(args: argparse.Namespace) -> int:
    settings = _settings()
    config_path = args.config
    info: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        model_cfg = TinyLMConfig.from_yaml(config_path)
        info["config"] = model_cfg.to_dict()
        info["analytic_parameters"] = model_cfg.estimate_parameters()
    checkpoint = args.checkpoint or settings["checkpoint"]
    if Path(checkpoint).exists():
        from runtime.inference import load_model

        loaded = load_model(checkpoint, args.tokenizer or settings["tokenizer"])
        info["checkpoint"] = loaded.info()
        info["measured_parameters"] = loaded.model.num_parameters()
    else:
        info["checkpoint"] = {"path": checkpoint, "exists": False}
    engine = ResearchEngine(db_path=settings.get("db_path", "storage/tinylm.db"))
    info["research"] = engine.stats()
    info["memory"] = MemoryStore(db_path=settings.get("db_path", "storage/tinylm.db")).stats()
    print(json.dumps(info, indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    settings = _settings()
    checkpoint = args.checkpoint or settings["checkpoint"]
    if _require_checkpoint(checkpoint) is None:
        return 1
    from runtime.inference import generate_text, load_model

    loaded = load_model(checkpoint, args.tokenizer or settings["tokenizer"], device=args.device)
    result = generate_text(
        loaded,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["text"])
        print(
            f"\n[{result['generated_tokens']} tokens, "
            f"{result['tokens_per_sec']:.1f} tok/s, {result['elapsed_sec']:.2f}s]",
            file=sys.stderr,
        )
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    settings = _settings()
    checkpoint = args.checkpoint or settings["checkpoint"]
    if _require_checkpoint(checkpoint) is None:
        return 1
    from agent.factory import build_agent
    from runtime.inference import load_model

    loaded = load_model(checkpoint, args.tokenizer or settings["tokenizer"], device=args.device)
    agent = build_agent(loaded.model, loaded.tokenizer, settings)
    print(f"TinyLM-50 chat (step {loaded.step}). /quit to exit, /tools to list tools.")
    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message in ("/quit", "/exit"):
            return 0
        if message == "/tools":
            print(agent.registry.describe())
            continue
        response = agent.chat(message)
        print(f"\nTinyLM-50: {response.text}")
        if response.tool_calls:
            print(
                "  tools: "
                + ", ".join(
                    f"{call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})"
                    f" -> {'ok' if call['ok'] else 'error'}"
                    for call in response.tool_calls
                ),
                file=sys.stderr,
            )
        if response.citations:
            for number, citation in enumerate(response.citations, start=1):
                print(f"  [{number}] {citation['title']} ({citation['path']})", file=sys.stderr)
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    settings = _settings()
    engine = ResearchEngine(
        db_path=args.db or settings.get("db_path", "storage/tinylm.db"),
        root=args.root or settings.get("document_root", "datasets/documents"),
    )
    if args.action == "ingest":
        if args.file:
            result = engine.ingest_file(args.file)
            print(json.dumps(result, indent=2))
        else:
            directory = args.dir or settings.get("document_root", "datasets/documents")
            result = engine.ingest_directory(directory, pattern=args.pattern)
            print(json.dumps(result, indent=2))
        return 0 if not result.get("failed") else 1
    if args.action == "search":
        query = args.query or args.file
        if not query:
            print("error: search needs a query", file=sys.stderr)
            return 1
        hits = engine.search(query, limit=args.limit)
        if args.json:
            print(json.dumps([hit.to_dict() for hit in hits], indent=2))
        else:
            for number, hit in enumerate(hits, start=1):
                print(f"[{number}] {hit.citation_text()}  score={hit.score:.3f}")
                print(hit.text[:600].replace("\n", " "))
                print()
        return 0
    if args.action == "list":
        print(json.dumps(engine.list_documents(), indent=2))
        return 0
    if args.action == "remove":
        target = args.path or args.file
        if not target:
            print("error: remove needs a document path", file=sys.stderr)
            return 1
        removed = engine.remove_document(target)
        print(json.dumps({"path": target, "removed": removed}))
        return 0 if removed else 1
    print(json.dumps(engine.stats(), indent=2))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    settings = _settings()
    store = MemoryStore(db_path=args.db or settings.get("db_path", "storage/tinylm.db"))
    if args.action == "save":
        entry = store.save(args.key, args.content, tags=args.tags, importance=args.importance)
        print(json.dumps(entry, indent=2))
    elif args.action == "get":
        entry = store.get(args.key)
        if entry is None:
            print(f"error: no memory with key {args.key!r}", file=sys.stderr)
            return 1
        print(json.dumps(entry, indent=2))
    elif args.action == "update":
        entry = store.update(
            args.key,
            content=args.content or None,
            tags=args.tags or None,
            importance=args.importance if args.importance != 0.5 else None,
        )
        print(json.dumps(entry, indent=2))
    elif args.action == "delete":
        print(json.dumps({"key": args.key, "deleted": store.delete(args.key)}))
    elif args.action == "search":
        print(json.dumps(store.search(args.query, limit=args.limit), indent=2))
    else:
        print(json.dumps(store.list(limit=args.limit), indent=2))
    return 0


def _default_registry() -> Any:
    from agent.factory import build_registry

    settings = _settings()
    registry, _engine, _memory = build_registry(
        db_path=settings.get("db_path", "storage/tinylm.db"),
        document_root=settings.get("document_root", "datasets/documents"),
    )
    return registry


def cmd_tools(args: argparse.Namespace) -> int:
    registry = _default_registry()
    if args.name:
        if args.name not in registry:
            print(f"error: unknown tool {args.name!r}; have {', '.join(registry.names())}", file=sys.stderr)
            return 1
        try:
            arguments = json.loads(args.arguments or "{}")
        except json.JSONDecodeError as exc:
            print(f"error: --arguments is not JSON: {exc.msg}", file=sys.stderr)
            return 1
        result = registry.execute(args.name, arguments, timeout=5.0)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 1
    print(json.dumps(registry.schemas(), indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    settings = _settings()
    checkpoint = args.checkpoint or settings["checkpoint"]
    if _require_checkpoint(checkpoint) is None:
        return 1
    from agent.factory import build_registry
    from evaluation.benchmark import benchmark
    from runtime.inference import load_model

    loaded = load_model(checkpoint, args.tokenizer or settings["tokenizer"], device=args.device)
    registry, engine, memory = build_registry(
        db_path=settings.get("db_path", "storage/tinylm.db"),
        document_root=settings.get("document_root", "datasets/documents"),
    )
    results = benchmark(
        loaded,
        registry,
        memory,
        engine,
        processed_dir=args.processed_dir,
        max_new_tokens=args.max_new_tokens,
        eval_batches=args.eval_batches,
        out_path=args.out,
    )
    print(json.dumps(results, indent=2))
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    settings = _settings()
    checkpoint = args.checkpoint or settings["checkpoint"]
    if _require_checkpoint(checkpoint) is None:
        return 1
    from evaluation.perplexity import evaluate_packed
    from data.pack import PackedDataset
    from runtime.inference import load_model
    from runtime.quantize import quantize_model, save_quantized

    loaded = load_model(checkpoint, args.tokenizer or settings["tokenizer"], device="cpu")
    valid_path = Path(args.processed_dir) / "valid_tokens.bin"
    packed = PackedDataset.load(valid_path) if valid_path.exists() else None
    before = (
        evaluate_packed(loaded.model, packed, max_batches=args.eval_batches, device="cpu", seq_len=256)
        if packed is not None
        else None
    )
    float32_weight_bytes = sum(
        parameter.numel() * 4 for parameter in loaded.model.parameters()
    )
    model, report = quantize_model(loaded.model, bits=args.bits)
    after = (
        evaluate_packed(model, packed, max_batches=args.eval_batches, device="cpu", seq_len=256)
        if packed is not None
        else None
    )
    suffix = "int8" if args.bits == 8 else "int4"
    out_path = Path(args.out or f"checkpoints/tiny50.{suffix}.pt")
    saved = save_quantized(out_path, model, loaded.config, report, step=loaded.step)
    payload = {
        "checkpoint": str(saved),
        "file_bytes": saved.stat().st_size,
        "quantization": report.as_dict(),
        # Training checkpoints also carry optimiser state; weight-only bytes are
        # the honest baseline for a quantisation comparison.
        "float32_weight_bytes": float32_weight_bytes,
        "training_checkpoint_bytes": Path(checkpoint).stat().st_size,
        "float32_to_file_ratio": float32_weight_bytes / saved.stat().st_size,
        "eval_before": before,
        "eval_after": after,
    }
    if before and after:
        payload["loss_delta"] = after["loss"] - before["loss"]
    report_path = saved.with_suffix(".json")
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinylm", description="TinyLM-50 command line")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show model, checkpoint, research and memory stats")
    info.add_argument("--checkpoint")
    info.add_argument("--tokenizer")
    info.set_defaults(func=cmd_info)

    generate = sub.add_parser("generate", help="generate text from a prompt")
    generate.add_argument("prompt")
    generate.add_argument("--checkpoint")
    generate.add_argument("--tokenizer")
    generate.add_argument("--max-new-tokens", type=int, default=None)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--repetition-penalty", type=float, default=1.1)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--json", action="store_true")
    generate.set_defaults(func=cmd_generate)

    chat = sub.add_parser("chat", help="interactive chat with tools")
    chat.add_argument("--checkpoint")
    chat.add_argument("--tokenizer")
    chat.add_argument("--device", default="auto")
    chat.set_defaults(func=cmd_chat)

    research = sub.add_parser("research", help="manage the offline research index")
    research.add_argument("action", choices=["ingest", "search", "list", "remove", "stats"])
    research.add_argument("file", nargs="?")
    research.add_argument("path", nargs="?")
    research.add_argument("--dir", default=None)
    research.add_argument("--pattern", default="**/*")
    research.add_argument("--query", default=None)
    research.add_argument("--limit", type=int, default=5)
    research.add_argument("--db", default=None)
    research.add_argument("--root", default=None)
    research.add_argument("--json", action="store_true")
    research.set_defaults(func=cmd_research)

    memory = sub.add_parser("memory", help="inspect and edit persistent memory")
    memory.add_argument("action", choices=["save", "get", "update", "delete", "search", "list"])
    memory.add_argument("key", nargs="?")
    memory.add_argument("content", nargs="?")
    memory.add_argument("--tags", default="")
    memory.add_argument("--importance", type=float, default=0.5)
    memory.add_argument("--limit", type=int, default=20)
    memory.add_argument("--query", default="")
    memory.add_argument("--db", default=None)
    memory.set_defaults(func=cmd_memory)

    tools = sub.add_parser("tools", help="list tool schemas or call one directly")
    tools.add_argument("name", nargs="?")
    tools.add_argument("--arguments", default="{}", help="JSON arguments for the tool")
    tools.set_defaults(func=cmd_tools)

    bench = sub.add_parser("benchmark", help="measure generation, perplexity and tools")
    bench.add_argument("--checkpoint")
    bench.add_argument("--tokenizer")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--processed-dir", default="datasets/processed")
    bench.add_argument("--max-new-tokens", type=int, default=32)
    bench.add_argument("--eval-batches", type=int, default=5)
    bench.add_argument("--out", default="storage/benchmark.json")
    bench.set_defaults(func=cmd_benchmark)

    quant = sub.add_parser("quantize", help="quantise weights to INT8 or INT4")
    quant.add_argument("--bits", type=int, choices=[4, 8], default=8)
    quant.add_argument("--checkpoint")
    quant.add_argument("--tokenizer")
    quant.add_argument("--processed-dir", default="datasets/processed")
    quant.add_argument("--eval-batches", type=int, default=5)
    quant.add_argument("--out", default=None)
    quant.set_defaults(func=cmd_quantize)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate" and args.max_new_tokens is None:
        args.max_new_tokens = int(_settings().get("max_new_tokens", 256))
    if args.command == "memory" and args.action in ("save", "get", "update", "delete") and not args.key:
        parser.error(f"memory {args.action} needs a key")
    if args.command == "memory" and args.action in ("save", "update") and args.action == "save" and not args.content:
        parser.error("memory save needs a key and content")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
