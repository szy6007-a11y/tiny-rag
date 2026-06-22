# Tiny RAG Agent TUI

Standalone TypeScript TUI for the Agent core loop.

This package is self-contained. The first build registers no tools, so it runs as a pure conversational Agent while keeping the same core shape:

- system prompt construction
- runtime context block
- multi-turn history replay
- streaming answer/thinking separation
- ReAct loop stop conditions
- OpenAI-compatible chat streaming
- function-calling protocol support with an empty registry

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

Optional:

```bash
npm start -- --context examples/context.json --history .agent-tui-history.jsonl
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

## Current Tool State

No tools are defined or registered in this build. The generic `ToolRegistry` and function-calling plumbing are present so concrete tools can be injected later without changing the Agent core.
