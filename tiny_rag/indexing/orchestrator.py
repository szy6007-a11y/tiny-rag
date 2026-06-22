from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from typing import Sequence

from tiny_rag.embedding import Embedder, OpenAIEmbedder
from tiny_rag.persistence.chunk_models import CHUNK_TYPE_TEXT

from .models import (
    CHUNK_SOURCE_TYPE,
    DEFAULT_RETRIEVER_TYPES,
    IndexInfo,
    IndexKnowledgeStats,
    VECTOR_RETRIEVER_TYPE,
)
from .repositories.base import IndexRepository
from .repositories.sqlite import SQLiteIndexRepository, connect_sqlite_index_db
from .service import KeywordsVectorHybridIndexer, normalize_retriever_types


DEFAULT_RETRIEVE_DRIVER = "sqlite"
SUPPORTED_RETRIEVE_DRIVERS = {"sqlite"}


def index_knowledge_after_chunks_persisted(
    *,
    knowledge_id: str,
    chunks: Sequence[object],
    conn: sqlite3.Connection | None = None,
    repository: IndexRepository | None = None,
    embedder: Embedder | None = None,
    title: str = "",
    retriever_types: Sequence[str] | None = DEFAULT_RETRIEVER_TYPES,
    knowledge_type: str = "",
) -> IndexKnowledgeStats:
    if chunks is None:
        raise ValueError(
            "chunks is required; pass PersistChunksResult.inserted_chunks so "
            "in-memory context_header is preserved for embedding content"
        )

    repo = repository or create_index_repository(conn)
    resolved_retriever_types = normalize_retriever_types(retriever_types)
    if VECTOR_RETRIEVER_TYPE in resolved_retriever_types and embedder is None:
        embedder = OpenAIEmbedder.from_env()

    dimension = embedder.get_dimensions() if embedder is not None else 0
    deleted_count = repo.delete_by_knowledge_id_list(
        [knowledge_id],
        dimension=dimension,
        knowledge_type=knowledge_type,
    )

    index_info_list = build_index_info_from_chunks(list(chunks), title=title)

    indexer = KeywordsVectorHybridIndexer(repo)
    estimated_storage_size = indexer.estimate_storage_size(
        embedder,
        index_info_list,
        resolved_retriever_types,
    )
    batch_stats = indexer.batch_index(
        embedder,
        index_info_list,
        resolved_retriever_types,
    )

    return IndexKnowledgeStats(
        metadata_count=batch_stats.metadata_count,
        fts_count=batch_stats.fts_count,
        vector_count=batch_stats.vector_count,
        skipped_duplicate_count=batch_stats.skipped_duplicate_count,
        batches=batch_stats.batches,
        requested_count=batch_stats.requested_count,
        indexed_count=batch_stats.indexed_count,
        deduplicated_count=batch_stats.deduplicated_count,
        embedded_count=batch_stats.embedded_count,
        knowledge_id=knowledge_id,
        deleted_count=deleted_count,
        text_chunk_count=len(index_info_list),
        estimated_storage_size=estimated_storage_size,
    )


def build_index_info_from_chunks(
    chunks: Sequence[object],
    *,
    title: str = "",
) -> list[IndexInfo]:
    title_prefix = ""
    stripped_title = title.strip()
    if stripped_title:
        title_prefix = stripped_title + "\n"

    index_info_list: list[IndexInfo] = []
    for chunk in chunks:
        row = getattr(chunk, "row", chunk)
        if _get(row, "chunk_type") != CHUNK_TYPE_TEXT:
            continue
        chunk_id = _get(row, "id")
        content = title_prefix + _embedding_content(
            content=_get(row, "content") or "",
            context_header=getattr(chunk, "context_header", "") or "",
        )
        index_info_list.append(
            IndexInfo(
                id=chunk_id,
                content=content,
                source_id=chunk_id,
                source_type=CHUNK_SOURCE_TYPE,
                chunk_id=chunk_id,
                knowledge_id=_get(row, "knowledge_id") or "",
                knowledge_base_id=_get(row, "knowledge_base_id") or "",
                tag_id=_get(row, "tag_id") or "",
                is_enabled=True,
            )
        )
    return index_info_list


def create_index_repository(conn: sqlite3.Connection | None = None) -> IndexRepository:
    driver = os.getenv("RETRIEVE_DRIVER", DEFAULT_RETRIEVE_DRIVER).strip().lower()
    if driver not in SUPPORTED_RETRIEVE_DRIVERS:
        raise ValueError(
            f"unsupported retrieve driver: {driver}; tiny-rag currently implements only "
            "sqlite."
        )
    return SQLiteIndexRepository(conn or _connect_from_env())


def _connect_from_env() -> sqlite3.Connection:
    db_path = os.getenv("TINY_RAG_DB_PATH", "./data/tiny-rag.sqlite")
    if db_path not in {":memory:", ""}:
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return connect_sqlite_index_db(db_path or ":memory:")


def _embedding_content(*, content: str, context_header: str) -> str:
    body = content.strip()
    if context_header == "":
        return body
    return context_header + "\n\n" + body


def _get(row: object, name: str):
    if isinstance(row, sqlite3.Row):
        return row[name]
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name)
