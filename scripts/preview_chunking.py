#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_rag.chunking import ParentChildResult, SplitterConfig, split_with_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview Tiny RAG chunking output.")
    parser.add_argument("path", help="UTF-8 text/Markdown file to chunk")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--separator", action="append", dest="separators", help="Override separators; repeatable")
    parser.add_argument("--language", action="append", dest="languages", default=[])
    parser.add_argument("--token-limit", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--show-full", action="store_true", help="Print full chunk content")
    parser.add_argument("--max-preview", type=int, default=160)
    args = parser.parse_args()

    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    cfg = SplitterConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=args.separators or [],
        strategy=args.strategy,
        token_limit=args.token_limit,
        languages=args.languages,
    )

    result, diagnostics = split_with_diagnostics(text, cfg)
    if isinstance(result, ParentChildResult):
        if args.json:
            print(json.dumps(_to_plain(result), ensure_ascii=False, indent=2))
        else:
            print(f"parents={len(result.parents)} children={len(result.children)}")
            print("\n# Parents")
            _print_chunks(result.parents, text, args)
            print("\n# Children")
            _print_chunks(result.children, text, args, include_parent=True)
        return 0

    chunks = result
    if args.json:
        print(
            json.dumps(
                {"diagnostics": _to_plain(diagnostics), "chunks": _to_plain(chunks)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"file={path}")
    print(f"chars={len(text)} chunks={len(chunks)}")
    print(f"selected_tier={diagnostics.selected_tier}")
    print(f"tier_chain={' -> '.join(diagnostics.tier_chain)}")
    if diagnostics.rejected:
        print("rejected=" + "; ".join(f"{item.tier}: {item.reason}" for item in diagnostics.rejected))
    print()
    _print_chunks(chunks, text, args)
    return 0


def _print_chunks(chunks, original_text: str, args, include_parent: bool = False) -> None:
    for chunk in chunks:
        parent = f" parent={chunk.parent_index}" if include_parent and hasattr(chunk, "parent_index") else ""
        print(f"[{chunk.seq}] range={chunk.start}:{chunk.end} len={len(chunk.content)}{parent}")
        if chunk.context_header:
            print("context:")
            print(_indent(chunk.context_header))
        print("content:")
        content = chunk.content if args.show_full else _preview(chunk.content, args.max_preview)
        print(_indent(content))
        if original_text[chunk.start : chunk.end] != chunk.content:
            print("WARNING: offset/content mismatch")
        print()


def _preview(text: str, max_chars: int) -> str:
    compact = text.replace("\r", "\\r")
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines() or [""])


def _to_plain(value):
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
