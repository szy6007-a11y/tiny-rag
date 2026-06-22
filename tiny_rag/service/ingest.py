from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Callable

from tiny_rag.converting.models.document import Document
from tiny_rag.converting.parser.docx_parser import DocxParser
from tiny_rag.converting.parser.markdown_parser import MarkdownParser
from tiny_rag.converting.parser.pdf_parser import PDFParser
from tiny_rag.embedding import Embedder, OpenAIEmbedder
from tiny_rag.indexing import index_knowledge_after_chunks_persisted
from tiny_rag.indexing.models import IndexKnowledgeStats, VECTOR_RETRIEVER_TYPE
from tiny_rag.indexing.repositories.base import IndexRepository
from tiny_rag.indexing.repositories.sqlite import SQLiteIndexRepository
from tiny_rag.persistence import ChunkRepository, persist_text_chunks
from tiny_rag.persistence.chunk_models import CHUNK_TYPE_TEXT

from .models import (
    IngestRequest,
    IngestResult,
    KnowledgeBaseConfig,
    KnowledgeRecord,
    default_knowledge_base_config,
)


ParserFactory = Callable[[str, str], object]


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
        self.embedder = embedder
        self.parser_factory = parser_factory or default_parser_factory
        self.knowledge_bases: dict[str, KnowledgeBaseConfig] = dict(knowledge_bases or {})
        self.knowledge_records: dict[str, KnowledgeRecord] = {}

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

        document = self._parse_document(payload, file_name=file_name, file_type=file_type)
        knowledge = KnowledgeRecord(
            id=request.knowledge_id,
            knowledge_base_id=kb.id,
            tenant_id=request.tenant_id,
            title=request.title or Path(file_name).stem or request.knowledge_id,
            file_name=file_name,
            file_type=file_type,
            metadata=dict(request.metadata or {}),
        )
        self.knowledge_records[knowledge.id] = knowledge

        persist_result = persist_text_chunks(
            self.conn,
            tenant_id=request.tenant_id,
            knowledge_id=knowledge.id,
            knowledge_base_id=kb.id,
            text=document.content,
            config=request.chunk_config,
        )

        retriever_types = tuple(
            request.retriever_types if request.retriever_types is not None else kb.retriever_types()
        )
        if not retriever_types:
            deleted_count = self.repository.delete_by_knowledge_id_list([knowledge.id])
            return IngestResult(
                knowledge=knowledge,
                knowledge_base=kb,
                document=document,
                persist_result=persist_result,
                index_stats=IndexKnowledgeStats(
                    knowledge_id=knowledge.id,
                    deleted_count=deleted_count,
                    text_chunk_count=sum(
                        1
                        for chunk in persist_result.inserted_chunks
                        if chunk.row.chunk_type == CHUNK_TYPE_TEXT
                    ),
                ),
            )

        embedder = self._embedder_for_indexing(retriever_types)
        try:
            index_stats = index_knowledge_after_chunks_persisted(
                conn=self.conn,
                repository=self.repository,
                embedder=embedder,
                knowledge_id=knowledge.id,
                title=knowledge.title,
                chunks=persist_result.inserted_chunks,
                retriever_types=retriever_types,
                knowledge_type=kb.kb_type,
            )
        except Exception:
            self._cleanup_failed_index(
                tenant_id=request.tenant_id,
                knowledge_id=knowledge.id,
                embedder=embedder,
                knowledge_type=kb.kb_type,
            )
            raise

        return IngestResult(
            knowledge=knowledge,
            knowledge_base=kb,
            document=document,
            persist_result=persist_result,
            index_stats=index_stats,
        )

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

    def _parse_document(self, content: bytes, *, file_name: str, file_type: str) -> Document:
        parser = self.parser_factory(file_name, file_type)
        return parser.parse_into_text(content)

    def _embedder_for_indexing(self, retriever_types: tuple[str, ...]) -> Embedder | None:
        if VECTOR_RETRIEVER_TYPE not in retriever_types:
            return self.embedder
        if self.embedder is not None:
            return self.embedder
        return OpenAIEmbedder.from_env()

    def _cleanup_failed_index(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        embedder: Embedder | None,
        knowledge_type: str,
    ) -> None:
        try:
            self.chunk_repository.soft_delete_by_knowledge_id(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
            )
        finally:
            dimension = 0
            if embedder is not None:
                try:
                    dimension = embedder.get_dimensions()
                except Exception:
                    dimension = 0
            self.repository.delete_by_knowledge_id_list(
                [knowledge_id],
                dimension=dimension,
                knowledge_type=knowledge_type,
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
