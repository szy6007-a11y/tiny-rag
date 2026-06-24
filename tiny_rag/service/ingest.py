from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
from typing import Callable, Sequence

from tiny_rag.chunking import Diagnostics, SplitterConfig
from tiny_rag.converting.models.document import Document
from tiny_rag.converting.parser.docx_parser import DocxParser
from tiny_rag.converting.parser.markdown_parser import MarkdownParser
from tiny_rag.converting.parser.pdf_parser import PDFParser
from tiny_rag.embedding import Embedder, OpenAIEmbedder
from tiny_rag.indexing import index_knowledge_after_chunks_persisted
from tiny_rag.indexing.models import IndexKnowledgeStats, VECTOR_RETRIEVER_TYPE
from tiny_rag.indexing.repositories.base import IndexRepository
from tiny_rag.indexing.repositories.sqlite import SQLiteIndexRepository
from tiny_rag.persistence import (
    ChunkRepository,
    PersistChunksResult,
    PersistChunksStats,
    persist_text_chunks,
)
from tiny_rag.persistence.chunk_models import CHUNK_TYPE_TEXT, utc_timestamp
from tiny_rag.persistence.sqlite import transaction

from .knowledge_lifecycle import (
    ENABLE_STATUS_DISABLED,
    ENABLE_STATUS_ENABLED,
    KNOWLEDGE_TYPE_FILE,
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_FINALIZING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
    KnowledgeRepository,
    replace_record,
)
from .models import (
    IngestRequest,
    IngestResult,
    KnowledgeBaseConfig,
    KnowledgeRecord,
    default_knowledge_base_config,
)


ParserFactory = Callable[[str, str], object]
TERMINAL_SKIP_STATUSES = {PARSE_STATUS_DELETING, PARSE_STATUS_CANCELLED}
CANCELLABLE_STATUSES = {
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
    PARSE_STATUS_FINALIZING,
}


class IngestService:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        repository: IndexRepository | None = None,
        embedder: Embedder | None = None,
        parser_factory: ParserFactory | None = None,
        knowledge_bases: dict[str, KnowledgeBaseConfig] | None = None,
    ) -> None:
        self.conn = conn
        self.repository = repository or SQLiteIndexRepository(conn)
        self.chunk_repository = ChunkRepository(conn)
        self.knowledge_repository = KnowledgeRepository(conn)
        self.embedder = embedder
        self.parser_factory = parser_factory or default_parser_factory
        self.knowledge_bases: dict[str, KnowledgeBaseConfig] = dict(knowledge_bases or {})
        self.knowledge_records: dict[str, KnowledgeRecord] = {}
        for kb in self.knowledge_bases.values():
            self.knowledge_records.update(
                self.knowledge_repository.active_records_map(
                    tenant_id=kb.tenant_id,
                    knowledge_base_id=kb.id,
                )
            )

    def ingest_document(self, request: IngestRequest) -> IngestResult:
        payload, file_name, file_type = _resolve_payload_and_name(request)
        kb = request.knowledge_base or self.knowledge_bases.get(request.knowledge_base_id)
        if kb is None:
            model_id = ""
            if self.embedder is not None:
                try:
                    model_id = self.embedder.get_model_id() or self.embedder.get_model_name()
                except Exception:
                    model_id = ""
            kb = default_knowledge_base_config(
                request.knowledge_base_id,
                tenant_id=request.tenant_id,
                embedding_model_id=model_id,
            )
        self.knowledge_bases[kb.id] = kb

        source = request.source
        if not source and isinstance(request.content, Path):
            source = str(request.content.expanduser().resolve())

        file_hash = content_hash(payload)
        file_size = len(payload)
        metadata = dict(request.metadata or {})
        if source:
            metadata.setdefault("path", source)

        existing = self._resolve_existing_knowledge(
            request=request,
            kb=kb,
            file_name=file_name,
            file_size=file_size,
            file_hash=file_hash,
            source=source,
        )
        if (
            existing is not None
            and not request.force
            and existing.file_hash == file_hash
            and existing.parse_status == PARSE_STATUS_COMPLETED
        ):
            self.knowledge_records[existing.id] = existing
            return self._skipped_result(existing, kb, "unchanged")

        knowledge_id = existing.id if existing is not None else request.knowledge_id
        if not knowledge_id:
            raise ValueError("knowledge_id is required")

        knowledge = KnowledgeRecord(
            id=knowledge_id,
            tenant_id=request.tenant_id,
            knowledge_base_id=kb.id,
            type=KNOWLEDGE_TYPE_FILE,
            title=request.title or Path(file_name).stem or knowledge_id,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_hash=file_hash,
            file_path=source,
            source=source,
            channel=request.channel or "web",
            parse_status=PARSE_STATUS_PENDING,
            enable_status=ENABLE_STATUS_DISABLED,
            embedding_model_id=kb.embedding_model_id,
            metadata=metadata,
            created_at=existing.created_at if existing is not None else "",
        )
        with transaction(self.conn):
            knowledge = self.knowledge_repository.upsert(knowledge)
        self.knowledge_records[knowledge.id] = knowledge

        return self.process_document(
            knowledge=knowledge,
            knowledge_base=kb,
            payload=payload,
            file_name=file_name,
            file_type=file_type,
            chunk_config=request.chunk_config,
            retriever_types=request.retriever_types,
            operation="updated" if existing is not None else "created",
        )

    def process_document(
        self,
        *,
        knowledge: KnowledgeRecord,
        knowledge_base: KnowledgeBaseConfig,
        payload: bytes,
        file_name: str,
        file_type: str,
        chunk_config: SplitterConfig | None = None,
        retriever_types: Sequence[str] | None = None,
        operation: str = "ingested",
    ) -> IngestResult:
        latest = self.knowledge_repository.get(
            tenant_id=knowledge.tenant_id,
            knowledge_id=knowledge.id,
        )
        if latest is not None:
            knowledge = latest

        if knowledge.parse_status in TERMINAL_SKIP_STATUSES:
            self.knowledge_records[knowledge.id] = knowledge
            return self._skipped_result(knowledge, knowledge_base, knowledge.parse_status)
        if knowledge.parse_status == PARSE_STATUS_COMPLETED:
            self.knowledge_records[knowledge.id] = knowledge
            return self._skipped_result(knowledge, knowledge_base, "already completed")

        knowledge = replace_record(
            knowledge,
            parse_status=PARSE_STATUS_PROCESSING,
            enable_status=ENABLE_STATUS_DISABLED,
            error_message="",
        )
        with transaction(self.conn):
            knowledge = self.knowledge_repository.update(knowledge)
        self.knowledge_records[knowledge.id] = knowledge

        resolved_retrievers = tuple(
            retriever_types if retriever_types is not None else knowledge_base.retriever_types()
        )
        embedder = self._embedder_for_indexing(resolved_retrievers)

        try:
            document = self._parse_document(payload, file_name=file_name, file_type=file_type)
            persist_result = persist_text_chunks(
                self.conn,
                tenant_id=knowledge.tenant_id,
                knowledge_id=knowledge.id,
                knowledge_base_id=knowledge_base.id,
                text=document.content,
                config=chunk_config,
            )
            index_stats = self._index_persisted_chunks(
                knowledge=knowledge,
                knowledge_base=knowledge_base,
                persist_result=persist_result,
                retriever_types=resolved_retrievers,
                embedder=embedder,
            )
        except Exception as exc:
            self._cleanup_failed_index(
                tenant_id=knowledge.tenant_id,
                knowledge_id=knowledge.id,
                embedder=embedder,
                knowledge_type=knowledge_base.kb_type,
            )
            failed = replace_record(
                knowledge,
                parse_status=PARSE_STATUS_FAILED,
                enable_status=ENABLE_STATUS_DISABLED,
                error_message=str(exc),
            )
            with transaction(self.conn):
                failed = self.knowledge_repository.update(failed)
            self.knowledge_records[knowledge.id] = failed
            raise

        completed = replace_record(
            knowledge,
            parse_status=PARSE_STATUS_COMPLETED,
            enable_status=ENABLE_STATUS_ENABLED,
            storage_size=index_stats.estimated_storage_size,
            processed_at=utc_timestamp(),
            error_message="",
        )
        with transaction(self.conn):
            completed = self.knowledge_repository.update(completed)
        self.knowledge_records[completed.id] = completed

        return IngestResult(
            knowledge=completed,
            knowledge_base=knowledge_base,
            document=document,
            persist_result=persist_result,
            index_stats=index_stats,
            operation=operation,
        )

    def reparse_knowledge(self, *, tenant_id: int, knowledge_id: str) -> IngestResult:
        knowledge = self.knowledge_repository.get(tenant_id=tenant_id, knowledge_id=knowledge_id)
        if knowledge is None:
            raise ValueError(f"knowledge not found: {knowledge_id}")
        if not knowledge.file_path:
            raise ValueError(f"knowledge has no file path for reparse: {knowledge_id}")
        path = Path(knowledge.file_path)
        if not path.exists():
            raise FileNotFoundError(f"knowledge source file not found: {path}")
        kb = self.knowledge_bases.get(knowledge.knowledge_base_id)
        if kb is None:
            kb = default_knowledge_base_config(
                knowledge.knowledge_base_id,
                tenant_id=knowledge.tenant_id,
                embedding_model_id=knowledge.embedding_model_id,
            )
            self.knowledge_bases[kb.id] = kb

        return self.ingest_document(
            IngestRequest(
                tenant_id=knowledge.tenant_id,
                knowledge_id=knowledge.id,
                knowledge_base_id=knowledge.knowledge_base_id,
                content=path,
                title=knowledge.title,
                file_name=knowledge.file_name,
                file_type=knowledge.file_type,
                knowledge_base=kb,
                metadata=knowledge.metadata,
                source=knowledge.source or knowledge.file_path,
                channel=knowledge.channel,
                force=True,
            )
        )

    def cancel_knowledge_parse(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
    ) -> KnowledgeRecord:
        knowledge = self.knowledge_repository.get(tenant_id=tenant_id, knowledge_id=knowledge_id)
        if knowledge is None:
            raise ValueError(f"knowledge not found: {knowledge_id}")
        if knowledge.parse_status == PARSE_STATUS_CANCELLED:
            return knowledge
        if knowledge.parse_status not in CANCELLABLE_STATUSES:
            raise ValueError(f"cannot cancel knowledge in status {knowledge.parse_status}")
        cancelled = replace_record(
            knowledge,
            parse_status=PARSE_STATUS_CANCELLED,
            enable_status=ENABLE_STATUS_DISABLED,
            error_message="cancelled by user",
        )
        with transaction(self.conn):
            cancelled = self.knowledge_repository.update(cancelled)
        self.knowledge_records[cancelled.id] = cancelled
        return cancelled

    def delete_knowledge(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
    ) -> KnowledgeRecord:
        knowledge = self.knowledge_repository.get(tenant_id=tenant_id, knowledge_id=knowledge_id)
        if knowledge is None:
            raise ValueError(f"knowledge not found: {knowledge_id}")
        deleting = replace_record(
            knowledge,
            parse_status=PARSE_STATUS_DELETING,
            enable_status=ENABLE_STATUS_DISABLED,
        )
        with transaction(self.conn):
            deleting = self.knowledge_repository.update(deleting)
        self._cleanup_knowledge_resources(deleting)
        with transaction(self.conn):
            self.knowledge_repository.soft_delete(tenant_id=tenant_id, knowledge_id=knowledge_id)
        self.knowledge_records.pop(knowledge_id, None)
        return replace_record(deleting, deleted_at=utc_timestamp())

    def ingest_text(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        knowledge_base_id: str,
        text: str,
        title: str = "",
        file_name: str = "document.md",
        knowledge_base: KnowledgeBaseConfig | None = None,
    ) -> IngestResult:
        return self.ingest_document(
            IngestRequest(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                knowledge_base_id=knowledge_base_id,
                content=text,
                title=title,
                file_name=file_name,
                file_type=Path(file_name).suffix.lstrip(".") or "md",
                knowledge_base=knowledge_base,
            )
        )

    def _resolve_existing_knowledge(
        self,
        *,
        request: IngestRequest,
        kb: KnowledgeBaseConfig,
        file_name: str,
        file_size: int,
        file_hash: str,
        source: str,
    ) -> KnowledgeRecord | None:
        existing = self.knowledge_repository.check_exists(
            tenant_id=request.tenant_id,
            knowledge_base_id=kb.id,
            knowledge_type=KNOWLEDGE_TYPE_FILE,
            file_hash=file_hash,
            file_name=file_name,
            file_size=file_size,
            source=source,
        )
        if existing is not None:
            return existing
        if request.knowledge_id:
            return self.knowledge_repository.get(
                tenant_id=request.tenant_id,
                knowledge_id=request.knowledge_id,
            )
        return None

    def _index_persisted_chunks(
        self,
        *,
        knowledge: KnowledgeRecord,
        knowledge_base: KnowledgeBaseConfig,
        persist_result: PersistChunksResult,
        retriever_types: tuple[str, ...],
        embedder: Embedder | None,
    ) -> IndexKnowledgeStats:
        if not retriever_types:
            deleted_count = self.repository.delete_by_knowledge_id_list([knowledge.id])
            return IndexKnowledgeStats(
                knowledge_id=knowledge.id,
                deleted_count=deleted_count,
                text_chunk_count=sum(
                    1
                    for chunk in persist_result.inserted_chunks
                    if chunk.row.chunk_type == CHUNK_TYPE_TEXT
                ),
            )

        return index_knowledge_after_chunks_persisted(
            conn=self.conn,
            repository=self.repository,
            embedder=embedder,
            knowledge_id=knowledge.id,
            title=knowledge.title,
            chunks=persist_result.inserted_chunks,
            retriever_types=retriever_types,
            knowledge_type=knowledge_base.kb_type,
        )

    def _parse_document(self, content: bytes, *, file_name: str, file_type: str) -> Document:
        parser = self.parser_factory(file_name, file_type)
        return parser.parse_into_text(content)

    def _embedder_for_indexing(self, retriever_types: tuple[str, ...]) -> Embedder | None:
        if VECTOR_RETRIEVER_TYPE not in retriever_types:
            return self.embedder
        if self.embedder is not None:
            return self.embedder
        return OpenAIEmbedder.from_env()

    def _cleanup_knowledge_resources(self, knowledge: KnowledgeRecord) -> None:
        kb = self.knowledge_bases.get(knowledge.knowledge_base_id)
        knowledge_type = kb.kb_type if kb is not None else ""
        embedder = self.embedder
        self._cleanup_failed_index(
            tenant_id=knowledge.tenant_id,
            knowledge_id=knowledge.id,
            embedder=embedder,
            knowledge_type=knowledge_type,
        )

    def _cleanup_failed_index(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        embedder: Embedder | None,
        knowledge_type: str,
    ) -> None:
        with transaction(self.conn):
            dimension = 0
            if embedder is not None:
                try:
                    dimension = embedder.get_dimensions()
                except Exception:
                    dimension = 0
            self.chunk_repository.soft_delete_by_knowledge_id(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
            )
            self.repository.delete_by_knowledge_id_list(
                [knowledge_id],
                dimension=dimension,
                knowledge_type=knowledge_type,
            )

    def _skipped_result(
        self,
        knowledge: KnowledgeRecord,
        knowledge_base: KnowledgeBaseConfig,
        message: str,
    ) -> IngestResult:
        text_count = len(
            self.chunk_repository.list_chunks_by_knowledge_id(
                tenant_id=knowledge.tenant_id,
                knowledge_id=knowledge.id,
            )
        )
        return IngestResult(
            knowledge=knowledge,
            knowledge_base=knowledge_base,
            document=Document(content=""),
            persist_result=PersistChunksResult(
                inserted_chunks=[],
                diagnostics=Diagnostics(),
                stats=PersistChunksStats(),
            ),
            index_stats=IndexKnowledgeStats(
                knowledge_id=knowledge.id,
                text_chunk_count=text_count,
            ),
            operation="skipped",
            skipped=True,
            message=message,
        )


def default_parser_factory(file_name: str, file_type: str):
    normalized = (file_type or Path(file_name).suffix.lstrip(".")).lower()
    if normalized in {"md", "markdown", "txt", "text", "json"}:
        return MarkdownParser(file_name=file_name, file_type=normalized)
    if normalized == "docx":
        return DocxParser(file_name=file_name, file_type=normalized)
    if normalized == "pdf":
        return PDFParser(file_name=file_name, file_type=normalized)
    raise ValueError(f"unsupported document type for ingest: {normalized or file_name}")


def content_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def _resolve_payload_and_name(request: IngestRequest) -> tuple[bytes, str, str]:
    content = request.content
    if isinstance(content, Path):
        payload = content.read_bytes()
        file_name = request.file_name or content.name
    elif isinstance(content, bytes):
        payload = content
        file_name = request.file_name or request.knowledge_id
    else:
        payload = str(content).encode("utf-8")
        file_name = request.file_name or request.knowledge_id + ".md"

    file_type = (request.file_type or Path(file_name).suffix.lstrip(".") or "md").lower()
    return payload, file_name, file_type
