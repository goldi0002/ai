"""Opt-in downloader for external pretraining corpora.

Nothing here runs automatically. Each source in ``datasets/sources.yaml`` must be
requested explicitly with ``--confirm``, and the licence is printed first so the
user can decide. Downloaded files land in ``datasets/pretraining/external/<name>``
and are recorded in ``datasets/sources_manifest.json`` for provenance.

Usage:
    python -m data.fetch --list
    python -m data.fetch --source gutenberg --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

SOURCES_FILE = Path("datasets/sources.yaml")
MANIFEST_FILE = Path("datasets/sources_manifest.json")
EXTERNAL_DIR = Path("datasets/pretraining/external")
MAX_BYTES = 200 * 1024 * 1024  # hard cap so an accidental URL cannot flood the disk
USER_AGENT = "TinyLM-50-dataset-fetch/1.0 (+offline research project)"


def load_sources() -> list[dict]:
    if not SOURCES_FILE.exists():
        raise FileNotFoundError(f"missing {SOURCES_FILE}")
    data = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8")) or {}
    return list(data.get("sources", []))


def _target_for(source: dict) -> Path:
    parsed = urllib.parse.urlparse(source["url"])
    name = Path(parsed.path).name or "download.txt"
    if "." not in name:
        name += ".txt"
    return EXTERNAL_DIR / source["name"] / name


def download(source: dict, max_bytes: int = MAX_BYTES) -> Path:
    target = _target_for(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"already present: {target} ({target.stat().st_size:,} bytes)")
        return target
    request = urllib.request.Request(source["url"], headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                out.close()
                target.unlink(missing_ok=True)
                raise RuntimeError(f"download exceeded {max_bytes} bytes; aborting")
            digest.update(block)
            out.write(block)
    print(f"downloaded {target} ({written:,} bytes)")
    _record(source, target, written, digest.hexdigest())
    return target


def _record(source: dict, target: Path, size: int, sha256: str) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"updated_at": None, "sources": []}
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest["sources"] = [s for s in manifest.get("sources", []) if s.get("name") != source["name"]]
    manifest["sources"].append(
        {
            "name": source["name"],
            "description": source.get("description", ""),
            "url": source["url"],
            "license": source.get("license", "unspecified"),
            "path": str(target),
            "bytes": size,
            "sha256": sha256,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "synthetic": False,
        }
    )
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in downloader for pretraining corpora")
    parser.add_argument("--list", action="store_true", help="show available sources")
    parser.add_argument("--source", help="name from datasets/sources.yaml")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="required acknowledgement that you reviewed the licence and want the download",
    )
    parser.add_argument("--max-mb", type=int, default=200)
    args = parser.parse_args(argv)

    sources = load_sources()
    if args.list or not args.source:
        print(f"{'name':<16}{'licence':<40}description")
        for source in sources:
            print(f"{source['name']:<16}{source.get('license', 'unspecified'):<40}{source.get('description', '')}")
        return 0

    match = next((s for s in sources if s["name"] == args.source), None)
    if match is None:
        print(f"unknown source {args.source!r}; use --list", file=sys.stderr)
        return 1
    print(f"source:    {match['name']}")
    print(f"url:       {match['url']}")
    print(f"licence:   {match.get('license', 'unspecified')}")
    if not args.confirm:
        print("refusing to download without --confirm (review the licence first)", file=sys.stderr)
        return 2
    try:
        download(match, max_bytes=args.max_mb * 1024 * 1024)
    except Exception as exc:  # network/disk errors are reported, not hidden
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print("Now run: python -m data.prepare prepare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
