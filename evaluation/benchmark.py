"""Benchmark actual, measured behaviour of a checkpoint and its tools.

Every number written by this module is measured in the current process; there
are no reference numbers baked in. Results are saved as JSON so they can be
quoted in the README with their machine and timestamp attached.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from agent.factory import build_registry
from data.pack import PackedDataset
from memory.store import MemoryStore
from tools.registry import ToolRegistry

from .perplexity import evaluate_packed


def _timed(fn, repeats: int = 5) -> Dict[str, float]:
    samples: List[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return {
        "runs": repeats,
        "mean_ms": statistics.mean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "min_ms": min(samples) * 1000,
        "max_ms": max(samples) * 1000,
    }


def benchmark(
    loaded,
    registry: ToolRegistry,
    memory: MemoryStore,
    engine,
    processed_dir: str | Path = "datasets/processed",
    max_new_tokens: int = 32,
    prompt: str = "Explain what a token is in one sentence.",
    eval_batches: int = 5,
    out_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    from runtime.inference import generate_text

    results: Dict[str, Any] = {
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "model": loaded.info(),
    }

    generation = generate_text(
        loaded,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    results["generation"] = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": generation["generated_tokens"],
        "elapsed_sec": generation["elapsed_sec"],
        "tokens_per_sec": generation["tokens_per_sec"],
    }

    valid_path = Path(processed_dir) / "valid_tokens.bin"
    if valid_path.exists():
        packed = PackedDataset.load(valid_path)
        results["perplexity"] = evaluate_packed(
            loaded.model,
            packed,
            batch_size=2,
            max_batches=eval_batches,
            device=loaded.device,
            seq_len=min(256, packed.seq_len),
        )
        results["perplexity"]["note"] = (
            f"{eval_batches} batches of 2 sequences, first "
            f"{min(256, packed.seq_len)} tokens per sequence"
        )
    else:
        results["perplexity"] = {"note": f"no packed validation data at {valid_path}"}

    cases = [
        ("calculator", {"expression": "(17 * 23 + 4) / 5"}),
        ("local_search", {"query": "BM25 ranking", "limit": 3}),
        ("retrieve_memory", {"query": "benchmark", "limit": 3}),
        ("list_files", {"pattern": "*.md", "max_results": 10}),
    ]
    tool_results = []
    for name, arguments in cases:
        def call(name=name, arguments=arguments):
            return registry.execute(name, arguments, timeout=5.0)

        first = call()
        timing = _timed(call, repeats=5)
        tool_results.append({"tool": name, "ok": first.ok, "latency": timing})
    results["tools"] = tool_results

    if engine is not None:
        hit = engine.search("quantisation", limit=3)
        timing = _timed(lambda: engine.search("quantisation", limit=3), repeats=5)
        results["research"] = {
            "documents": engine.stats()["documents"],
            "chunks": engine.stats()["chunks"],
            "query": "quantisation",
            "hits": len(hit),
            "latency": timing,
        }

    memory.save("benchmark_probe", "temporary benchmark note", tags="benchmark", importance=0.1)
    timing = _timed(lambda: memory.search("benchmark", 3), repeats=5)
    results["memory"] = {"memories": memory.stats()["memories"], "latency": timing}
    memory.delete("benchmark_probe")

    checkpoint = Path(loaded.checkpoint_path)
    results["checkpoint"] = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
    }
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv=None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    from runtime.inference import load_model

    parser = argparse.ArgumentParser(description="Benchmark a TinyLM-50 checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer/tiny50_sp.model")
    parser.add_argument("--tools", action="store_true", help="benchmark tools and research search")
    parser.add_argument("--out", default="storage/benchmark.json")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)

    loaded = load_model(args.checkpoint, args.tokenizer)
    registry, engine, memory = build_registry(
        db_path="storage/tinylm.db", document_root="datasets/documents"
    )
    if not args.tools:
        engine = None
    results = benchmark(
        loaded,
        registry,
        memory,
        engine,
        max_new_tokens=args.max_new_tokens,
        out_path=args.out,
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
