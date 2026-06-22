# Tiny RAG

Lightweight document converting, chunking, persistence, and indexing utilities
for local RAG experiments.

The project currently has two parts:

- `tiny_rag/`: Python library code for document parsing, chunking, chunk
  persistence, embedding, indexing, and retrieval.
- `agent-tui/`: standalone TypeScript terminal chat agent with an
  OpenAI-compatible streaming client and function-calling plumbing. No RAG tools
  are registered yet.

## Pipeline

The Python path is organized as:

```text
document bytes
  -> converting parser
  -> normalized text / Markdown
  -> chunking
  -> SQLite chunk persistence
  -> keyword and vector indexing
  -> retrieval
```

Important entrypoints:

- `tiny_rag.chunking.split_with_diagnostics`: split normalized text into chunks
  with strategy diagnostics.
- `tiny_rag.persistence.persist_text_chunks`: persist text chunks into the
  `chunks` table while preserving in-memory context headers for indexing.
- `tiny_rag.indexing.index_knowledge_after_chunks_persisted`: build index
  records from persisted chunks, delete stale index rows for the knowledge item,
  and write keyword/vector index data.
- `tiny_rag.indexing.repositories.sqlite.SQLiteIndexRepository`: SQLite-backed
  keyword retrieval through FTS5 and vector retrieval through `sqlite-vec`.

Minimal in-memory flow:

```python
from tiny_rag.chunking import SplitterConfig
from tiny_rag.indexing import index_knowledge_after_chunks_persisted
from tiny_rag.indexing.models import RetrieveParams
from tiny_rag.indexing.repositories.sqlite import SQLiteIndexRepository
from tiny_rag.persistence import connect, persist_text_chunks

conn = connect(":memory:")
persisted = persist_text_chunks(
    conn,
    tenant_id=1,
    knowledge_id="knowledge-1",
    knowledge_base_id="kb-1",
    text="# Runbook\n\nThe latency_budget is 120ms.",
    config=SplitterConfig(strategy="auto"),
)

repository = SQLiteIndexRepository(conn)
stats = index_knowledge_after_chunks_persisted(
    conn=conn,
    repository=repository,
    knowledge_id="knowledge-1",
    title="Runbook",
    chunks=persisted.inserted_chunks,
    retriever_types=["keywords"],
)

results = repository.retrieve(
    RetrieveParams(query="latency_budget", knowledge_ids=["knowledge-1"], top_k=3)
)
```

For vector indexing, pass a custom embedder or configure the OpenAI-compatible
embedder with:

```text
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

## Tests

Use the project test runner through `uv`:

```bash
uv run python scripts/run_tests.py unit
uv run python scripts/run_tests.py integration
uv run python scripts/run_tests.py full
```

The default `unit` lane covers chunking, persistence, embedding batch behavior,
and SQLite indexing/retrieval behavior. The `integration` lane exercises parser
smoke tests and an in-memory chunk persistence plus keyword/vector indexing flow.

Optional parser dependencies for the integration lane can be installed with:

```bash
uv sync --group integration
```

## Agent TUI

The TUI lives in `agent-tui/` and is intentionally self-contained:

```bash
cd agent-tui
npm install
npm run verify:prompts
npm start
```

It loads `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL_NAME` from environment
variables or a `.env` found by walking upward from the current directory. See
`agent-tui/README.md` for commands, context loading, and prompt verification.

Current status: the agent supports streaming responses, thinking/answer
separation, persisted local history, and generic function-calling execution, but
the tool registry is empty. Hooking it up to the Python retrieval pipeline is the
next integration step.
