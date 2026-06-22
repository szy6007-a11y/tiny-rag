# Test Lanes

Use `scripts/run_tests.py` through `uv run` as the canonical test entrypoint.

```bash
uv run python scripts/run_tests.py unit
uv run python scripts/run_tests.py integration
uv run python scripts/run_tests.py full
```

`unit` is the default lane and has no optional document-conversion dependencies.
It covers chunking behavior, offsets, routing diagnostics, parent-child chunking,
language markers, table context, protected spans, chunk persistence, embedding
batch behavior, and SQLite indexing/retrieval behavior.

`integration` runs the converting-to-chunking smoke pipeline for Markdown, text,
DOCX, and PDF. It also exercises an in-memory Markdown ingest through chunk
persistence, keyword indexing, vector indexing, and retrieval. Parser smoke tests
generate documents in a temporary directory and skip cleanly when optional parser
dependencies are not installed.

The historical command remains available while callers migrate:

```bash
uv run python -m unittest tests.test_chunking
```

For manual inspection of real text fixtures, use:

```bash
uv run python scripts/preview_chunking.py tests/fixtures/documents/markdown/sample_manual.md
```

Install optional parser dependencies for the integration lane with:

```bash
uv sync --group integration
```
