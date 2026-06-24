#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from scripts.tui_rag_bridge import (
    DEFAULT_DIMENSIONS,
    DEFAULT_KB_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_TUI_ALLOWED_TOOLS,
    LocalHashEmbedder,
    LocalLexicalReranker,
    manifest_path,
    tui_doc_payload,
    tui_kb_payload,
    tui_tool_definitions,
)
from tiny_rag.agent import AgentConfig
from tiny_rag.chunking import SplitterConfig
from tiny_rag.persistence import connect
from tiny_rag.persistence.sqlite import transaction
from tiny_rag.retrieval import RerankService
from tiny_rag.service.agent import AgentService
from tiny_rag.service.ingest import IngestService
from tiny_rag.service.knowledge_lifecycle import PARSE_STATUS_COMPLETED
from tiny_rag.service.models import IngestRequest, KnowledgeBaseConfig, KnowledgeRecord


ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = ROOT / "knowledge"
DEFAULT_DB_PATH = ROOT / ".agent-tui-rag.sqlite"
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".json", ".docx", ".pdf"}


def ingest_documents(
    *,
    db: str | Path,
    documents: Sequence[str | Path],
    knowledge_ids: Sequence[str] = (),
    knowledge_base_id: str = DEFAULT_KB_ID,
    knowledge_base_name: str = "Local TUI Knowledge Base",
    tenant_id: int = DEFAULT_TENANT_ID,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    embedding_dimensions: int = DEFAULT_DIMENSIONS,
    max_tool_output_chars: int = 16000,
    allowed_tools: Sequence[str] = (),
) -> dict[str, Any]:
    documents = [Path(item).expanduser().resolve() for item in documents]
    missing = [str(path) for path in documents if not path.exists()]
    if missing:
        raise FileNotFoundError("document not found: " + ", ".join(missing))

    db_path = Path(db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(db_path)

    tenant_id = int(tenant_id)
    allowed_tools = list(allowed_tools or DEFAULT_TUI_ALLOWED_TOOLS)
    embedder = LocalHashEmbedder(int(embedding_dimensions))
    kb = KnowledgeBaseConfig(
        id=knowledge_base_id,
        tenant_id=tenant_id,
        name=knowledge_base_name or "Local TUI Knowledge Base",
        embedding_model_id=embedder.get_model_id(),
        vector_enabled=True,
        keyword_enabled=True,
    )

    conn = connect(db_path)
    try:
        ingest_service = IngestService(conn, embedder=embedder, knowledge_bases={kb.id: kb})
        _bootstrap_manifest_records(ingest_service, mpath, kb, tenant_id)

        current_paths = {str(path) for path in documents}
        existing_by_source = {
            record.source: record
            for record in ingest_service.knowledge_repository.list_by_knowledge_base(
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
            )
            if record.source
        }
        for index, path in enumerate(documents, start=1):
            source = str(path)
            existing = existing_by_source.get(source)
            knowledge_id = (
                existing.id
                if existing is not None
                else knowledge_ids[index - 1]
                if index - 1 < len(knowledge_ids)
                else stable_knowledge_id(kb.id, path)
            )
            result = ingest_service.ingest_document(
                IngestRequest(
                    tenant_id=tenant_id,
                    knowledge_id=knowledge_id,
                    knowledge_base_id=kb.id,
                    title=path.stem,
                    file_name=path.name,
                    file_type=path.suffix.lstrip(".").lower() or "md",
                    content=path,
                    chunk_config=SplitterConfig(
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        strategy="auto",
                    ),
                    knowledge_base=kb,
                    source=source,
                )
            )

        for record in list(
            ingest_service.knowledge_repository.list_by_knowledge_base(
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
            )
        ):
            if record.source and record.source not in current_paths:
                ingest_service.delete_knowledge(tenant_id=tenant_id, knowledge_id=record.id)

        manifest_docs = [
            manifest_doc_from_record(ingest_service, record)
            for record in ingest_service.knowledge_repository.list_by_knowledge_base(
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
            )
            if record.parse_status == PARSE_STATUS_COMPLETED
        ]

        service = AgentService(
            conn,
            repository=ingest_service.repository,
            chunk_repository=ingest_service.chunk_repository,
            embedder=embedder,
            knowledge_bases=ingest_service.knowledge_bases,
            knowledge_records=ingest_service.knowledge_records,
            rerank_service=RerankService(LocalLexicalReranker()),
        )
        config = AgentConfig(
            allowed_tools=allowed_tools,
            knowledge_bases=[kb.id],
            knowledge_ids=[doc["knowledge_id"] for doc in manifest_docs],
            max_tool_output_chars=int(max_tool_output_chars),
        )
        registry = service.create_tool_registry(config, tenant_id=tenant_id)
        tools = tui_tool_definitions(registry)
    finally:
        conn.close()

    manifest = {
        "tenant_id": tenant_id,
        "db_path": str(db_path),
        "embedding_dimensions": int(embedding_dimensions),
        "max_tool_output_chars": int(max_tool_output_chars),
        "allowed_tools": allowed_tools,
        "knowledge_base": asdict(kb),
        "documents": manifest_docs,
    }
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "success": True,
        "db_path": str(db_path),
        "manifest_path": str(mpath),
        "knowledge_base": tui_kb_payload(kb, manifest_docs),
        "selected_documents": [tui_doc_payload(doc) for doc in manifest_docs],
        "tools": tools,
    }


def sync_documents_dir(
    *,
    docs_dir: Path,
    db: str | Path,
    knowledge_base_id: str = DEFAULT_KB_ID,
    knowledge_base_name: str = "Local TUI Knowledge Base",
    tenant_id: int = DEFAULT_TENANT_ID,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    embedding_dimensions: int = DEFAULT_DIMENSIONS,
    max_tool_output_chars: int = 16000,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not docs_dir.exists():
        if not allow_empty:
            raise FileNotFoundError(f"knowledge directory not found: {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        documents: list[Path] = []
    else:
        documents = discover_documents(docs_dir)

    if not documents and not allow_empty:
        raise FileNotFoundError(
            f"no supported documents found in {docs_dir}; supported extensions: "
            + ", ".join(sorted(SUPPORTED_SUFFIXES))
        )

    return ingest_documents(
        db=db,
        documents=documents,
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=knowledge_base_name,
        tenant_id=tenant_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_dimensions=embedding_dimensions,
        max_tool_output_chars=max_tool_output_chars,
    )


def stable_knowledge_id(knowledge_base_id: str, path: Path) -> str:
    digest = hashlib.sha1(f"{knowledge_base_id}:{path}".encode("utf-8")).hexdigest()
    return f"local-doc-{digest[:16]}"


def manifest_doc_from_record(service: IngestService, record: KnowledgeRecord) -> dict[str, Any]:
    chunk_count = len(
        service.chunk_repository.list_chunks_by_knowledge_id(
            tenant_id=record.tenant_id,
            knowledge_id=record.id,
        )
    )
    return {
        "knowledge_id": record.id,
        "knowledge_base_id": record.knowledge_base_id,
        "title": record.title,
        "file_name": record.file_name,
        "file_type": record.file_type,
        "path": record.source or record.file_path,
        "file_size": record.file_size,
        "file_hash": record.file_hash,
        "parse_status": record.parse_status,
        "chunk_count": chunk_count,
    }


def _bootstrap_manifest_records(
    service: IngestService,
    mpath: Path,
    kb: KnowledgeBaseConfig,
    tenant_id: int,
) -> None:
    if service.knowledge_repository.list_by_knowledge_base(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
    ):
        return
    if not mpath.exists():
        return
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return
    with transaction(service.conn):
        for doc in manifest.get("documents") or []:
            knowledge_id = str(doc.get("knowledge_id") or "")
            if not knowledge_id:
                continue
            source = str(doc.get("path") or "")
            record = KnowledgeRecord(
                id=knowledge_id,
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
                title=str(doc.get("title") or doc.get("file_name") or knowledge_id),
                file_name=str(doc.get("file_name") or ""),
                file_type=str(doc.get("file_type") or ""),
                file_size=int(doc.get("file_size") or 0),
                file_hash=str(doc.get("file_hash") or ""),
                source=source,
                file_path=source,
                parse_status=PARSE_STATUS_COMPLETED,
                enable_status="enabled",
                embedding_model_id=kb.embedding_model_id,
                metadata={"path": source} if source else {},
            )
            service.knowledge_repository.upsert(record)
            service.knowledge_records[record.id] = record


def discover_documents(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"knowledge directory not found: {docs_dir}")
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"knowledge path is not a directory: {docs_dir}")
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(docs_dir).parts)
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def document_signature(docs_dir: Path) -> tuple[tuple[str, int, int], ...]:
    if not docs_dir.exists() or not docs_dir.is_dir():
        return ()
    signature: list[tuple[str, int, int]] = []
    for path in discover_documents(docs_dir):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signature.append((str(path), int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def print_sync_result(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return
    docs = payload.get("selected_documents", [])
    print(
        f"Tiny RAG knowledge sync complete. Documents indexed: {len(docs)}",
        flush=True,
    )


def sync_from_args(args: argparse.Namespace, *, allow_empty: bool) -> dict[str, Any]:
    return sync_documents_dir(
        docs_dir=Path(args.docs_dir).expanduser().resolve(),
        db=Path(args.db).expanduser().resolve(),
        knowledge_base_id=args.knowledge_base_id,
        knowledge_base_name=args.knowledge_base_name,
        tenant_id=args.tenant_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_dimensions=args.embedding_dimensions,
        max_tool_output_chars=args.max_tool_output_chars,
        allow_empty=allow_empty,
    )


def watch_documents(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir).expanduser().resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)
    interval = max(0.5, float(args.watch_interval))
    last_signature: tuple[tuple[str, int, int], ...] | None = None

    if not args.json:
        print(f"Watching knowledge directory: {docs_dir}", flush=True)
    try:
        while True:
            signature = document_signature(docs_dir)
            if signature != last_signature:
                try:
                    payload = sync_from_args(args, allow_empty=True)
                except Exception as exc:
                    if args.json:
                        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), flush=True)
                    else:
                        print(f"Knowledge sync failed: {exc}", file=sys.stderr, flush=True)
                else:
                    last_signature = signature
                    print_sync_result(payload, as_json=args.json)
            time.sleep(interval)
    except KeyboardInterrupt:
        if not args.json:
            print("Stopped watching knowledge directory.", flush=True)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the offline Tiny RAG TUI index.")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="Document library directory.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    parser.add_argument("--knowledge-base-id", default=DEFAULT_KB_ID)
    parser.add_argument("--knowledge-base-name", default="Local TUI Knowledge Base")
    parser.add_argument("--tenant-id", type=int, default=DEFAULT_TENANT_ID)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--embedding-dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--max-tool-output-chars", type=int, default=16000)
    parser.add_argument("--json", action="store_true", help="Print the ingest result as JSON.")
    parser.add_argument("--allow-empty", action="store_true", help="Write an empty index when no documents exist.")
    parser.add_argument("--watch", action="store_true", help="Continuously sync the knowledge directory.")
    parser.add_argument("--watch-interval", type=float, default=1.5, help="Polling interval for --watch.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.watch:
        return watch_documents(args)

    try:
        payload = sync_from_args(args, allow_empty=args.allow_empty)
    except Exception as exc:
        if args.json:
            print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    doc_titles = [
        doc.get("title") or doc.get("file_name") or doc.get("knowledge_id")
        for doc in payload.get("selected_documents", [])
    ]
    print("Tiny RAG offline ingest complete.")
    print(f"Documents indexed: {len(doc_titles)}")
    print(f"Database: {payload['db_path']}")
    print(f"Manifest: {payload['manifest_path']}")
    if doc_titles:
        print("Indexed documents:")
        for title in doc_titles:
            print(f"- {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
