# Tiny RAG Agent TUI

Standalone TypeScript TUI for the Agent core loop.

This package is the online chat surface. It starts in Agent Q&A mode, opens an
existing Tiny RAG SQLite index plus manifest, and registers the core RAG tools
in the TypeScript function-calling loop. At startup it syncs the repository
root `knowledge/` directory into that index, then watches for document changes
while the TUI is running.

- system prompt construction
- runtime context block
- multi-turn history replay
- streaming answer/thinking separation
- ReAct loop stop conditions
- OpenAI-compatible chat streaming
- function-calling protocol support
- local RAG tools backed by Python Tiny RAG

The prompt template is vendored locally:

```text
prompts/agent_system_prompt.yaml
```

## Run

From the repository root:

```bash
npm run dev
```

Place local documents under `knowledge/`. Adding, editing, or deleting a
supported file while the TUI is running automatically updates the local index.

From this package directly:

```bash
cd agent-tui
npm install
npm run verify:prompts
npm start
```

This opens the TUI, syncs the root `knowledge/` directory, and registers RAG
tools against `.agent-tui-rag.sqlite`.

Optional context/history override:

```bash
npm start -- --context examples/context.json --history .agent-tui-history.jsonl
```

Start pure chat mode without a KB:

```bash
npm start -- --no-rag
```

Disable automatic knowledge-folder sync and use an already-built index:

```bash
npm start -- --no-rag-sync
```

## Environment

By default the TUI searches upward for `tiny-rag/.env` and uses the same env file as the rest of tiny-rag:

```bash
LLM_MODEL_NAME=deepseek-v4-pro
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=...
```

You can override with `--model`, `--base-url`, `--api-key`, or `--env`.

When running from outside the repo, pass the env path explicitly:

```bash
npm start -- --env /Users/shenzhengyang/workspace/tiny-rag/.env
```

## Prompt Verification

```bash
npm run verify:prompts
```

This checks:

- runtime `pure` / `rag` prompt strings are byte-for-byte identical to the corresponding YAML `content: |` blocks

## TUI Commands

```text
/help
/clear
/clear-history
/model
/context
/quit
```

Long transcript output can be reviewed with the mouse wheel, `Up`/`Down`,
`PageUp`/`PageDown`, `Home`, and `End`. When a long assistant response
finishes, the viewport jumps to the start of that response so the first half is
immediately readable.

## RAG Tool Mode

When RAG is enabled, the TUI registers the core RAG tools that are wired
through the Python service/tool registry:

- `knowledge_search`
- `grep_chunks`

The bridge uses the existing Python service/tool registry and a local hash
embedder plus lexical reranker, so document retrieval does not require separate
embedding or rerank API keys. `knowledge_search` handles semantic retrieval;
`grep_chunks` performs exact regex/literal lookup over indexed chunk content.
Other Python-side helper tools such as `list_knowledge_chunks` and
`get_document_info` remain available in the service layer until they are wired
into the default TUI Agent surface.

The TUI does not rebuild the whole knowledge base on every change. Its startup
sync and watcher call the repository-root `ingest.py`, which keeps stable
knowledge IDs, skips unchanged file hashes, and replaces only the changed
document's chunks and indexes.
