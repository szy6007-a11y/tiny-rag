#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_rag.agent import AgentConfig
from tiny_rag.agent.tools import ToolGrepChunks, ToolKnowledgeSearch
from tiny_rag.persistence import connect
from tiny_rag.retrieval import RankResult, RerankService
from tiny_rag.service.agent import AgentService
from tiny_rag.service.knowledge_lifecycle import KnowledgeRepository, PARSE_STATUS_COMPLETED
from tiny_rag.service.models import KnowledgeBaseConfig, KnowledgeRecord


MANIFEST_SUFFIX = ".manifest.json"
DEFAULT_KB_ID = "local-tui-kb"
DEFAULT_TENANT_ID = 1
DEFAULT_DIMENSIONS = 128
DEFAULT_TUI_ALLOWED_TOOLS = (ToolKnowledgeSearch, ToolGrepChunks)
TUI_KNOWLEDGE_SEARCH_DESCRIPTION = """Search the bound local knowledge base and return complete evidence chunks for answering the user.

Use this as the semantic retrieval tool in the TUI Agent. For exact identifiers,
error codes, names, or literal strings, use grep_chunks when it is available.
The local pipeline combines vector-style matching, keyword retrieval, reranking,
and context merging.

Input:
- queries: 1-5 short search questions or phrases that express what evidence is
  needed.
- knowledge_base_ids: optional scope filter; omit unless the runtime context
  lists multiple bound knowledge bases.

Output:
Returns ranked chunks with document names, scores, snippets, and full chunk
content. After receiving results, answer from the returned content. If the
results do not contain enough evidence, try one refined search or tell the user
the knowledge base does not contain enough information.

Do not request helper tools that are not in the active tool list."""


class LocalHashEmbedder:
    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed(self, text: str, **_kwargs: Any) -> list[float]:
        return self._vector(text)

    def batch_embed(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def get_model_name(self) -> str:
        return "tiny-rag-local-hash"

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_model_id(self) -> str:
        return f"{self.get_model_name()}:{self.dimensions}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vec[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vec))
        if norm <= 0:
            return vec
        return [value / norm for value in vec]


class LocalLexicalReranker:
    def rerank(self, query: str, passages: Sequence[str], **_kwargs: Any) -> list[RankResult]:
        query_tokens = set(tokenize(query))
        ranks: list[RankResult] = []
        for index, passage in enumerate(passages):
            passage_tokens = set(tokenize(passage))
            if not query_tokens or not passage_tokens:
                score = 0.0
            else:
                overlap = len(query_tokens & passage_tokens)
                coverage = overlap / max(1, len(query_tokens))
                precision = overlap / max(1, len(passage_tokens))
                score = min(1.0, 0.85 * coverage + 0.15 * math.sqrt(precision))
            ranks.append(RankResult(index=index, relevance_score=score))
        ranks.sort(key=lambda item: item.relevance_score, reverse=True)
        return ranks


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    raw = [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]
    tokens: list[str] = []
    cjk_run: list[str] = []

    def flush_cjk() -> None:
        if not cjk_run:
            return
        tokens.extend(cjk_run)
        if len(cjk_run) > 1:
            tokens.extend(cjk_run[index] + cjk_run[index + 1] for index in range(len(cjk_run) - 1))
        cjk_run.clear()

    for token in raw:
        if len(token) == 1 and "\u3400" <= token <= "\u9fff":
            cjk_run.append(token)
            continue
        flush_cjk()
        tokens.append(token)
    flush_cjk()
    return tokens


def manifest_path(db_path: str | Path) -> Path:
    return Path(str(db_path) + MANIFEST_SUFFIX)


def load_manifest(db_path: str | Path) -> dict[str, Any]:
    path = manifest_path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"RAG manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")


def tool_result_to_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {"success": False, "output": "", "error": f"unsupported tool result: {type(result).__name__}"}
    payload.setdefault("success", False)
    payload.setdefault("output", "")
    return payload


def tui_tool_definitions(registry) -> list[dict[str, Any]]:
    tools = []
    for definition in registry.get_function_definitions():
        tool = definition.to_chat_tool()
        if tool.get("function", {}).get("name") == ToolKnowledgeSearch:
            tool["function"]["description"] = TUI_KNOWLEDGE_SEARCH_DESCRIPTION
        tools.append(tool)
    return tools


def build_services(db_path: str | Path, manifest: dict[str, Any]) -> tuple[AgentService, int, AgentConfig]:
    tenant_id = int(manifest.get("tenant_id") or DEFAULT_TENANT_ID)
    embedder = LocalHashEmbedder(int(manifest.get("embedding_dimensions") or DEFAULT_DIMENSIONS))
    kb_data = manifest.get("knowledge_base") or {}
    kb = KnowledgeBaseConfig(
        id=str(kb_data.get("id") or DEFAULT_KB_ID),
        tenant_id=tenant_id,
        name=str(kb_data.get("name") or "Local TUI Knowledge Base"),
        embedding_model_id=embedder.get_model_id(),
        vector_enabled=True,
        keyword_enabled=True,
    )
    conn = connect(db_path)
    service = AgentService(
        conn,
        embedder=embedder,
        knowledge_bases={kb.id: kb},
        rerank_service=RerankService(LocalLexicalReranker()),
    )
    records = records_from_db(conn, kb.id, tenant_id)
    if not records:
        records = {}
        for doc in manifest.get("documents") or []:
            knowledge_id = str(doc.get("knowledge_id") or "")
            if not knowledge_id:
                continue
            records[knowledge_id] = service_record_from_manifest(doc, kb.id, tenant_id)
    service.knowledge_records.update(records)
    config = AgentConfig(
        allowed_tools=allowed_tools_from_manifest(manifest),
        knowledge_bases=[kb.id],
        knowledge_ids=list(records),
        max_tool_output_chars=int(manifest.get("max_tool_output_chars") or 16000),
    )
    return service, tenant_id, config


def service_record_from_manifest(doc: dict[str, Any], kb_id: str, tenant_id: int) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=str(doc.get("knowledge_id") or ""),
        knowledge_base_id=kb_id,
        tenant_id=tenant_id,
        title=str(doc.get("title") or doc.get("file_name") or doc.get("knowledge_id") or ""),
        file_name=str(doc.get("file_name") or ""),
        file_type=str(doc.get("file_type") or ""),
        file_size=int(doc.get("file_size") or 0),
        file_hash=str(doc.get("file_hash") or ""),
        source=str(doc.get("path") or ""),
        file_path=str(doc.get("path") or ""),
        metadata={"path": str(doc.get("path") or "")},
    )


def records_from_db(conn, kb_id: str, tenant_id: int) -> dict[str, KnowledgeRecord]:
    try:
        repo = KnowledgeRepository(conn)
        records = repo.list_by_knowledge_base(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
        )
    except Exception:
        return {}
    return {
        record.id: record
        for record in records
        if record.parse_status == PARSE_STATUS_COMPLETED
    }


def docs_from_records(service: AgentService, records: Sequence[KnowledgeRecord]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (item.created_at, item.id)):
        chunk_count = len(
            service.chunk_repository.list_chunks_by_knowledge_id(
                tenant_id=record.tenant_id,
                knowledge_id=record.id,
            )
        )
        docs.append(
            {
                "knowledge_id": record.id,
                "knowledge_base_id": record.knowledge_base_id,
                "title": record.title,
                "file_name": record.file_name,
                "file_type": record.file_type,
                "file_size": record.file_size,
                "file_hash": record.file_hash,
                "path": record.source or record.file_path,
                "chunk_count": chunk_count,
            }
        )
    return docs


def allowed_tools_from_manifest(manifest: dict[str, Any]) -> list[str]:
    raw = manifest.get("allowed_tools") or DEFAULT_TUI_ALLOWED_TOOLS
    if not isinstance(raw, (list, tuple)):
        return list(DEFAULT_TUI_ALLOWED_TOOLS)
    tools = [str(item).strip() for item in raw if str(item).strip()]
    return tools or list(DEFAULT_TUI_ALLOWED_TOOLS)


def execute_command(args: argparse.Namespace) -> int:
    raw = sys.stdin.read()
    tool_args = json.loads(raw or "{}")
    manifest = load_manifest(args.db)
    service, tenant_id, config = build_services(args.db, manifest)
    try:
        registry = service.create_tool_registry(config, tenant_id=tenant_id)
        result = registry.execute_tool(args.tool, tool_args)
        write_json(tool_result_to_payload(result))
    finally:
        service.conn.close()
    return 0


def definitions_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.db)
    service, tenant_id, config = build_services(args.db, manifest)
    try:
        registry = service.create_tool_registry(config, tenant_id=tenant_id)
        docs = docs_from_records(service, service.knowledge_records.values())
        if not docs:
            docs = list(manifest.get("documents") or [])
        write_json(
            {
                "success": True,
                "tools": tui_tool_definitions(registry),
                "knowledge_base": tui_kb_payload(
                    KnowledgeBaseConfig(
                        id=str(manifest["knowledge_base"]["id"]),
                        tenant_id=int(manifest.get("tenant_id") or DEFAULT_TENANT_ID),
                        name=str(manifest["knowledge_base"].get("name") or "Local TUI Knowledge Base"),
                    ),
                    docs,
                ),
                "selected_documents": [tui_doc_payload(doc) for doc in docs],
            }
        )
    finally:
        service.conn.close()
    return 0


def tui_kb_payload(kb: KnowledgeBaseConfig, docs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": kb.id,
        "name": kb.name or kb.id,
        "type": "document",
        "description": "Local documents indexed by the Tiny RAG TUI bridge.",
        "doc_count": len(docs),
        "capabilities": ["keyword", "vector", "rerank", "chunk_read"],
        "recent_docs": [
            {
                "knowledge_id": str(doc.get("knowledge_id") or ""),
                "title": str(doc.get("title") or ""),
                "file_name": str(doc.get("file_name") or ""),
                "file_size": int(doc.get("file_size") or 0),
                "type": str(doc.get("file_type") or ""),
                "description": f"{int(doc.get('chunk_count') or 0)} chunks indexed",
            }
            for doc in docs
        ],
    }


def tui_doc_payload(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "knowledge_id": str(doc.get("knowledge_id") or ""),
        "knowledge_base_id": str(doc.get("knowledge_base_id") or DEFAULT_KB_ID),
        "title": str(doc.get("title") or doc.get("file_name") or ""),
        "file_name": str(doc.get("file_name") or ""),
        "file_type": str(doc.get("file_type") or ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny RAG TUI Python bridge")
    sub = parser.add_subparsers(dest="command", required=True)

    execute = sub.add_parser("execute", help="execute one registered RAG tool")
    execute.add_argument("--db", required=True)
    execute.add_argument("--tool", required=True)
    execute.set_defaults(func=execute_command)

    definitions = sub.add_parser("definitions", help="print registered tool definitions")
    definitions.add_argument("--db", required=True)
    definitions.set_defaults(func=definitions_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        write_json({"success": False, "output": "", "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
