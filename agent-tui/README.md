# Tiny RAG Agent TUI

Standalone TypeScript TUI for the Agent core loop.

This package is self-contained. By default it starts in Agent Q&A mode with a
local knowledge base already bound. On startup it indexes the bundled default KB
through the Python Tiny RAG stack and registers the core RAG tool in the
TypeScript function-calling loop.

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

```bash
cd agent-tui
npm install
npm run verify:prompts
npm start
```

This opens the TUI with the default KB and the core RAG tool registered.

Optional context/history override:

```bash
npm start -- --context examples/context.json --history .agent-tui-history.jsonl
```

Use your own local document(s) instead of the bundled default KB:

```bash
npm run dev -- \
  --rag-document /path/to/manual.md
```

Repeat `--rag-document` to bind multiple local files. Startup rebuilds the
local SQLite index at `.agent-tui-rag.sqlite` by default.

Start pure chat mode without a KB:

```bash
npm start -- --no-rag
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

## RAG Tool Mode

By default, the TUI registers the core RAG tools that are wired through the
Python service/tool registry:

- `knowledge_search`
- `grep_chunks`

The bridge uses the existing Python service/tool registry and a local hash
embedder plus lexical reranker, so document retrieval does not require separate
embedding or rerank API keys. `knowledge_search` handles semantic retrieval;
`grep_chunks` performs exact regex/literal lookup over indexed chunk content.
Other Python-side helper tools such as `list_knowledge_chunks` and
`get_document_info` remain available in the service layer until they are wired
into the default TUI Agent surface.
