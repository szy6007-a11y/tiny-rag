# AGENTS.md

## Scope

These instructions apply to the `tiny_rag.chunking` package.

## Module Contract

- Keep the public API exported from `__init__.py` stable unless the caller-facing contract is intentionally changed.
- Preserve exact source offsets: for every emitted chunk, `original_text[chunk.start:chunk.end] == chunk.content`.
- Treat `context_header` as retrieval context only. It must not be counted in `start`/`end` offsets.
- Keep `Chunk.embedding_content()` as the single place that combines `context_header` with body content.
- Normalize `SplitterConfig` before strategy logic, and preserve the existing distinction between public default overlap and direct splitter calls that allow zero overlap.

## Strategy Notes

- `split_with_diagnostics()` owns strategy selection, fallback, validation, and diagnostics.
- `auto` strategy should profile the document, then try semantic tiers before falling back to `legacy`.
- `heading` is for Markdown heading structure and should fall back to `legacy` when the structure is too sparse.
- `heuristic` is for PDF-like or plain text structure markers such as form feeds, chapter markers, page footers, visual separators, and blank blocks.
- `legacy` is the final compatibility splitter and should remain conservative.
- Avoid introducing boundaries inside protected spans such as fenced code blocks and similar protected regions.

## Testing

- Run `uv run python scripts/run_tests.py unit` after behavioral changes.
- Run `uv run python scripts/run_tests.py integration` when touching converting-to-chunking handoff behavior; this lane skips cleanly if optional document parser dependencies are not installed.
- Add or update tests for offset integrity, diagnostics/fallback behavior, parent-child chunking, language-specific markers, table handling, and protected spans when those areas change.
- Use `scripts/preview_chunking.py` for manual inspection of real Markdown or text fixtures.
