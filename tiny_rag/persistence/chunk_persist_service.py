from __future__ import annotations

import sqlite3
from typing import Iterable

from tiny_rag.chunking import ParentChildResult, SplitterConfig, split_with_diagnostics
from tiny_rag.chunking.types import ChildChunk, Chunk

from .chunk_models import (
    CHUNK_TYPE_PARENT_TEXT,
    CHUNK_TYPE_TEXT,
    ChunkRow,
    PersistedChunk,
    PersistChunksResult,
    PersistChunksStats,
)
from .chunk_repository import ChunkRepository
from .schema import ensure_chunks_schema
from .sqlite import transaction


def persist_text_chunks(
    conn: sqlite3.Connection,
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    text: str,
    config: SplitterConfig | None = None,
) -> PersistChunksResult:
    split_result, diagnostics = split_with_diagnostics(text, config or SplitterConfig())
    persisted_chunks, skipped_empty_count = _build_persisted_chunks(
        split_result,
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
    )
    rows = [item.row for item in persisted_chunks]

    ensure_chunks_schema(conn)
    repo = ChunkRepository(conn)
    with transaction(conn):
        deleted_count = repo.soft_delete_by_knowledge_id(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
        )
        repo.create_chunks(rows, batch_size=100)

    stats = PersistChunksStats(
        inserted_count=len(persisted_chunks),
        text_count=sum(1 for item in persisted_chunks if item.row.chunk_type == CHUNK_TYPE_TEXT),
        parent_count=sum(
            1 for item in persisted_chunks if item.row.chunk_type == CHUNK_TYPE_PARENT_TEXT
        ),
        skipped_empty_count=skipped_empty_count,
        deleted_count=deleted_count,
    )
    return PersistChunksResult(
        inserted_chunks=persisted_chunks,
        diagnostics=diagnostics,
        stats=stats,
    )


def _build_persisted_chunks(
    split_result,
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
) -> tuple[list[PersistedChunk], int]:
    if isinstance(split_result, ParentChildResult):
        return _build_parent_child_chunks(
            split_result,
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
        )

    return _build_flat_chunks(
        split_result,
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
    )


def _build_flat_chunks(
    chunks: Iterable[Chunk],
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
) -> tuple[list[PersistedChunk], int]:
    persisted_chunks: list[PersistedChunk] = []
    skipped_empty_count = 0
    for chunk in chunks:
        if chunk.content.strip() == "":
            skipped_empty_count += 1
            continue
        persisted_chunks.append(
            PersistedChunk(
                row=_new_row(
                    chunk,
                    tenant_id=tenant_id,
                    knowledge_id=knowledge_id,
                    knowledge_base_id=knowledge_base_id,
                    chunk_type=CHUNK_TYPE_TEXT,
                ),
                context_header=chunk.context_header,
            )
        )

    persisted_chunks.sort(key=lambda item: item.row.chunk_index)
    _link_prev_next([item.row for item in persisted_chunks])
    return persisted_chunks, skipped_empty_count


def _build_parent_child_chunks(
    result: ParentChildResult,
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
) -> tuple[list[PersistedChunk], int]:
    parent_chunks = [
        PersistedChunk(
            row=_new_row(
                parent,
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                knowledge_base_id=knowledge_base_id,
                chunk_type=CHUNK_TYPE_PARENT_TEXT,
            ),
            context_header=parent.context_header,
        )
        for parent in result.parents
    ]
    parent_rows = [item.row for item in parent_chunks]
    _link_prev_next(parent_rows)

    has_parent_child = len(parent_rows) > 0
    persisted_chunks: list[PersistedChunk] = list(parent_chunks)
    skipped_empty_count = 0
    for child in result.children:
        if child.content.strip() == "":
            skipped_empty_count += 1
            continue

        row = _new_row(
            child,
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            chunk_type=CHUNK_TYPE_TEXT,
        )
        if 0 <= child.parent_index < len(parent_rows):
            row.parent_chunk_id = parent_rows[child.parent_index].id
        persisted_chunks.append(PersistedChunk(row=row, context_header=child.context_header))

    persisted_chunks.sort(key=lambda item: item.row.chunk_index)
    if not has_parent_child:
        _link_prev_next(
            [item.row for item in persisted_chunks if item.row.chunk_type == CHUNK_TYPE_TEXT]
        )
    return persisted_chunks, skipped_empty_count


def _new_row(
    chunk: Chunk | ChildChunk,
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    chunk_type: str,
) -> ChunkRow:
    return ChunkRow(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=chunk.content,
        chunk_index=chunk.seq,
        start_at=chunk.start,
        end_at=chunk.end,
        chunk_type=chunk_type,
    )


def _link_prev_next(rows: list[ChunkRow]) -> None:
    for index, row in enumerate(rows):
        if index > 0:
            rows[index - 1].next_chunk_id = row.id
            row.pre_chunk_id = rows[index - 1].id
