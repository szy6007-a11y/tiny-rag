# Tiny RAG Default Knowledge Base

## Agent Application

Tiny RAG is a local-first RAG Agent application skeleton. It connects document
conversion, structure-aware chunking, SQLite persistence, hybrid retrieval,
reranking, context merging, and an OpenAI-compatible function-calling Agent
loop.

The TypeScript TUI starts in Agent Q&A mode by default. On startup it indexes
this default knowledge base and registers the core Python Tiny RAG tools:

- `knowledge_search` for semantic retrieval
- `grep_chunks` for exact keyword/regex lookup

Other Python-side tools exist for development and service-level workflows.
`list_knowledge_chunks` and `get_document_info` remain service-layer tools until
they are wired into the default TUI Agent surface.

## Retrieval Pipeline

The local RAG path is:

```text
document
  -> parser
  -> chunker
  -> SQLite chunk store
  -> FTS5 keyword index + sqlite-vec vector index
  -> hybrid retrieval + rerank + merge
  -> RAG tools
  -> Agent answer
```

The TUI bridge uses local hash embeddings and a lexical reranker so retrieval
does not require a separate embedding or rerank API key. The chat model still
uses the configured OpenAI-compatible LLM.

## Usage

Start the TUI from `agent-tui/`:

```bash
npm start
```

The app enters Agent Q&A mode immediately. You can ask questions such as:

- What tools does the Agent use for knowledge retrieval?
- What is the local RAG pipeline?
- Why does the TUI bridge not need a separate embedding API key?

To bind your own documents, pass `--rag-document` one or more times. To start a
pure conversational shell without RAG, pass `--no-rag`.
