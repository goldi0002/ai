"""Dataset preparation CLI for TinyLM-50.

Subcommands
-----------
seed      Build the deterministic synthetic seed corpus (tokenizer/smoke only).
prepare   Clean, filter, deduplicate and split all pretraining text.
tokenize  Train the SentencePiece BPE tokenizer (16k).
pack      Tokenize packed uint16 files for pretraining/validation.
manifest  Print the recorded dataset manifest.

Typical offline run:
    python -m data.prepare seed
    python -m data.prepare prepare
    python -m data.prepare tokenize
    python -m data.prepare pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

from data.clean import clean_documents
from data.deduplicate import deduplicate
from data.pack import pack_documents, split_documents
from data.seed_corpus import write_corpus

DEFAULT_PRETRAIN_DIR = Path("datasets/pretraining")
DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_TOKENIZER_PREFIX = Path("data/tokenizer/tiny50_sp")
TEXT_SUFFIXES = {".txt", ".md", ".jsonl", ".json"}


def read_text_documents(root: Path) -> List[Tuple[str, str]]:
    """Read every supported file under ``root`` as (doc_id, text).

    JSONL files are flattened into one document per line so that prose datasets
    and instruction files can share the loader without being conflated.
    """
    documents: List[Tuple[str, str]] = []
    if not root.exists():
        return documents
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_id = str(path.relative_to(root))
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                documents.append((f"{doc_id}#{line_number}", line))
        elif path.suffix.lower() == ".json" and path.name.endswith("manifest.json"):
            continue
        else:
            documents.append((doc_id, text))
    return documents


def _license_index(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        entry["name"]: entry
        for entry in data.get("sources", [])
        if isinstance(entry, dict) and "name" in entry
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_prepare(args: argparse.Namespace) -> int:
    root = Path(args.input_dir)
    documents = read_text_documents(root)
    if not documents:
        print(f"error: no documents found under {root}", file=sys.stderr)
        return 1

    licenses = _license_index(Path("datasets/sources_manifest.json"))
    cleaned, clean_stats = clean_documents(
        documents, min_chars=args.min_chars, min_words=args.min_words
    )
    deduped, dedup_stats = deduplicate(cleaned, threshold=args.threshold)
    if not deduped:
        print("error: every document was filtered out", file=sys.stderr)
        return 1

    train_docs, valid_docs = split_documents(deduped, valid_ratio=args.valid_ratio, seed=args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, docs in (("pretrain_train.jsonl", train_docs), ("pretrain_valid.jsonl", valid_docs)):
        with (out_dir / name).open("w", encoding="utf-8") as handle:
            for doc_id, text in docs:
                handle.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")

    sources = sorted({doc_id.split("/")[0] for doc_id, _ in deduped})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": ["normalize", "quality_filter", "deduplicate", "document_split"],
        "input_dir": str(root),
        "synthetic": root.name == "seed" or "seed" in root.name,
        "sources": sources,
        "licenses": {
            source: licenses.get(source, {"license": "unspecified", "synthetic": True})
            for source in sources
        },
        "documents": {
            "input": clean_stats["input_documents"],
            "after_clean": clean_stats["kept_documents"],
            "after_dedup": len(deduped),
            "train": len(train_docs),
            "valid": len(valid_docs),
        },
        "clean_stats": clean_stats,
        "dedup_stats": dedup_stats,
        "train_chars": sum(len(t) for _, t in train_docs),
        "valid_chars": sum(len(t) for _, t in valid_docs),
        "train_sha256": _sha256(out_dir / "pretrain_train.jsonl"),
        "valid_sha256": _sha256(out_dir / "pretrain_valid.jsonl"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["documents"], indent=2))
    print(f"train chars: {manifest['train_chars']:,}  valid chars: {manifest['valid_chars']:,}")
    print(f"manifest: {out_dir / 'manifest.json'}")
    return 0


def _read_processed(path: Path) -> List[Tuple[str, str]]:
    documents = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            documents.append((record["id"], record["text"]))
    return documents


def cmd_tokenize(args: argparse.Namespace) -> int:
    from model.tokenizer import TinyTokenizer

    processed = Path(args.processed_dir)
    train_path = processed / "pretrain_train.jsonl"
    if not train_path.exists():
        print("error: run `python -m data.prepare prepare` first", file=sys.stderr)
        return 1
    # SentencePiece drops lines longer than max_sentence_length, and JSONL
    # wraps whole documents onto one long line. Train on extracted plain text.
    corpus_path = processed / "tokenizer_corpus.txt"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for split in ("train", "valid"):
            path = processed / f"pretrain_{split}.jsonl"
            if path.exists():
                for _, text in _read_processed(path):
                    handle.write(text.strip() + "\n\n")
    from data.seed_corpus import write_diversity_corpus

    diversity_path = write_diversity_corpus(processed / "tokenizer_diversity.txt")
    corpus_paths = [corpus_path, diversity_path]

    prefix = Path(args.model_prefix)
    tokenizer = TinyTokenizer.train(
        corpus_paths,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )
    print(f"tokenizer: {prefix}.model  vocab={tokenizer.vocab_size}")
    print(f"special ids: {json.dumps(tokenizer.special_ids)}")
    meta_path = prefix.with_name(prefix.name + "_meta.json")
    print(f"metadata: {meta_path}")
    if tokenizer.vocab_size != args.vocab_size:
        print(
            f"error: trained vocab {tokenizer.vocab_size} != requested {args.vocab_size}",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    from model.tokenizer import TinyTokenizer

    processed = Path(args.processed_dir)
    tokenizer = TinyTokenizer(args.tokenizer)
    total = 0
    for split in ("train", "valid"):
        path = processed / f"pretrain_{split}.jsonl"
        if not path.exists():
            print(f"error: missing {path}", file=sys.stderr)
            return 1
        documents = _read_processed(path)
        packed = pack_documents(
            documents,
            tokenizer,
            out_prefix=processed / f"{split}_tokens",
            seq_len=args.seq_len,
        )
        total += packed.num_tokens
        print(
            f"{split}: {packed.num_sequences} sequences x {packed.seq_len} "
            f"= {packed.num_tokens:,} tokens -> {packed.bin_path}"
        )
    (processed / "pack_meta.json").write_text(
        json.dumps({"seq_len": args.seq_len, "total_tokens": total}, indent=2), encoding="utf-8"
    )
    print(f"total tokens: {total:,}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    paths = write_corpus(args.out_dir, target_kb=args.target_kb, seed=args.seed)
    total = sum(path.stat().st_size for path in paths)
    for path in paths:
        print(f"wrote {path}")
    print(f"total {total / 1024:.1f} KiB")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    path = Path(args.output_dir) / "manifest.json"
    if not path.exists():
        print("error: no manifest yet; run prepare first", file=sys.stderr)
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyLM-50 data preparation")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="generate the synthetic seed corpus")
    seed.add_argument("--out-dir", default=str(DEFAULT_PRETRAIN_DIR / "seed_generated"))
    seed.add_argument("--target-kb", type=int, default=1500)
    seed.add_argument("--seed", type=int, default=1337)
    seed.set_defaults(func=cmd_seed)

    prepare = sub.add_parser("prepare", help="clean, deduplicate and split text")
    prepare.add_argument("--input-dir", default=str(DEFAULT_PRETRAIN_DIR))
    prepare.add_argument("--output-dir", default=str(DEFAULT_PROCESSED_DIR))
    prepare.add_argument("--valid-ratio", type=float, default=0.02)
    prepare.add_argument("--min-chars", type=int, default=200)
    prepare.add_argument("--min-words", type=int, default=30)
    prepare.add_argument("--threshold", type=int, default=3, help="SimHash Hamming threshold")
    prepare.add_argument("--seed", type=int, default=1337)
    prepare.set_defaults(func=cmd_prepare)

    tok = sub.add_parser("tokenize", help="train the SentencePiece tokenizer")
    tok.add_argument("--model-prefix", default=str(DEFAULT_TOKENIZER_PREFIX))
    tok.add_argument("--vocab-size", type=int, default=16000)
    tok.add_argument("--character-coverage", type=float, default=0.9995)
    tok.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    tok.set_defaults(func=cmd_tokenize)

    pack = sub.add_parser("pack", help="tokenize and pack binary training files")
    pack.add_argument("--tokenizer", default="data/tokenizer/tiny50_sp.model")
    pack.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    pack.add_argument("--seq-len", type=int, default=1024)
    pack.set_defaults(func=cmd_pack)

    man = sub.add_parser("manifest", help="print the dataset manifest")
    man.add_argument("--output-dir", default=str(DEFAULT_PROCESSED_DIR))
    man.set_defaults(func=cmd_manifest)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
