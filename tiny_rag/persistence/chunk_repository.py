from __future__ import annotations

import json
import sqlite3
from typing import Iterable, Sequence

from .chunk_models import CHUNK_TYPE_PARENT_TEXT, CHUNK_TYPE_TEXT, ChunkRow, utc_timestamp


INSERT_COLUMNS = [
    "id",
    "seq_id",
    "tenant_id",
    "knowledge_base_id",
    "knowledge_id",
    "content",
    "chunk_index",
    "is_enabled",
    "start_at",
    "end_at",
    "pre_chunk_id",
    "next_chunk_id",
    "chunk_type",
    "parent_chunk_id",
    "image_info",
    "video_info",
    "relation_chunks",
    "indirect_relation_chunks",
    "metadata",
    "tag_id",
    "status",
    "content_hash",
    "flags",
    "created_at",
    "updated_at",
    "deleted_at",
]


def clean_invalid_utf8(value: str) -> str:
    cleaned = value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return cleaned.replace("\x00", "")


class ChunkRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def soft_delete_by_knowledge_id(self, *, tenant_id: int, knowledge_id: str) -> int:
        cursor = self.conn.execute(
            """
            UPDATE chunks
            SET deleted_at = ?
            WHERE tenant_id = ?
              AND knowledge_id = ?
              AND deleted_at IS NULL
            """,
            (utc_timestamp(), tenant_id, knowledge_id),
        )
        return cursor.rowcount

    def create_chunks(self, chunks: Iterable[ChunkRow], *, batch_size: int = 100) -> list[ChunkRow]:
        rows = list(chunks)
        if not rows:
            return rows

        for row in rows:
            row.content = clean_invalid_utf8(row.content)

        self.assign_seq_ids(rows)
        placeholders = ", ".join("?" for _ in INSERT_COLUMNS)
        columns = ", ".join(INSERT_COLUMNS)
        sql = f"INSERT INTO chunks ({columns}) VALUES ({placeholders})"

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            self.conn.executemany(sql, [self._to_values(row) for row in batch])

        return rows

    def assign_seq_ids(self, chunks: Sequence[ChunkRow]) -> None:
        if not any(row.seq_id in (None, 0) for row in chunks):
            return

        max_seq_id = self.conn.execute("SELECT MAX(seq_id) FROM chunks").fetchone()[0]
        next_seq_id = 1 if max_seq_id is None else int(max_seq_id) + 1
        for row in chunks:
            if row.seq_id in (None, 0):
                row.seq_id = next_seq_id
                next_seq_id += 1

    def list_chunks_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        include_deleted: bool = False,
    ) -> list[ChunkRow]:
        return self._list_chunks_by_knowledge_id(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            include_deleted=include_deleted,
            chunk_type=CHUNK_TYPE_TEXT,
        )

    def list_chunks_by_knowledge_base_id(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        include_deleted: bool = False,
    ) -> list[ChunkRow]:
        return self._list_chunks_by_knowledge_base_id(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            include_deleted=include_deleted,
            chunk_type=CHUNK_TYPE_TEXT,
        )

    def list_parent_chunks_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        include_deleted: bool = False,
    ) -> list[ChunkRow]:
        return self._list_chunks_by_knowledge_id(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            include_deleted=include_deleted,
            chunk_type=CHUNK_TYPE_PARENT_TEXT,
        )

    def list_all_chunks_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        include_deleted: bool = False,
    ) -> list[ChunkRow]:
        return self._list_chunks_by_knowledge_id(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            include_deleted=include_deleted,
        )

    def _list_chunks_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        include_deleted: bool,
        chunk_type: str | None = None,
    ) -> list[ChunkRow]:
        where_deleted = "" if include_deleted else "AND deleted_at IS NULL"
        type_filter = "" if chunk_type is None else "AND chunk_type = ?"
        params: tuple = (tenant_id, knowledge_id)
        if chunk_type is not None:
            params = (tenant_id, knowledge_id, chunk_type)
        rows = self.conn.execute(
            f"""
            SELECT {", ".join(INSERT_COLUMNS)}
            FROM chunks
            WHERE tenant_id = ?
              AND knowledge_id = ?
              {type_filter}
              {where_deleted}
            ORDER BY chunk_index ASC, seq_id ASC
            """,
            params,
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def _list_chunks_by_knowledge_base_id(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        include_deleted: bool,
        chunk_type: str | None = None,
    ) -> list[ChunkRow]:
        where_deleted = "" if include_deleted else "AND deleted_at IS NULL"
        type_filter = "" if chunk_type is None else "AND chunk_type = ?"
        params: tuple = (tenant_id, knowledge_base_id)
        if chunk_type is not None:
            params = (tenant_id, knowledge_base_id, chunk_type)
        rows = self.conn.execute(
            f"""
            SELECT {", ".join(INSERT_COLUMNS)}
            FROM chunks
            WHERE tenant_id = ?
              AND knowledge_base_id = ?
              {type_filter}
              {where_deleted}
            ORDER BY knowledge_id ASC, chunk_index ASC, seq_id ASC
            """,
            params,
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def list_chunks_by_id(
        self,
        *,
        tenant_id: int,
        chunk_ids: Sequence[str],
        include_deleted: bool = False,
    ) -> list[ChunkRow]:
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        where_deleted = "" if include_deleted else "AND deleted_at IS NULL"
        rows = self.conn.execute(
            f"""
            SELECT {", ".join(INSERT_COLUMNS)}
            FROM chunks
            WHERE tenant_id = ?
              AND id IN ({placeholders})
              {where_deleted}
            """,
            (tenant_id, *ids),
        ).fetchall()
        chunk_map = {row["id"]: self._from_row(row) for row in rows}
        return [chunk_map[chunk_id] for chunk_id in ids if chunk_id in chunk_map]

    def list_chunks_by_parent_ids(
        self,
        *,
        tenant_id: int,
        parent_ids: Sequence[str],
        include_deleted: bool = False,
    ) -> list[ChunkRow]:
        ids = [parent_id for parent_id in parent_ids if parent_id]
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        where_deleted = "" if include_deleted else "AND deleted_at IS NULL"
        rows = self.conn.execute(
            f"""
            SELECT {", ".join(INSERT_COLUMNS)}
            FROM chunks
            WHERE tenant_id = ?
              AND parent_chunk_id IN ({placeholders})
              {where_deleted}
            ORDER BY chunk_index ASC, seq_id ASC
            """,
            (tenant_id, *ids),
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def _to_values(self, row: ChunkRow) -> tuple:
        return (
            row.id,
            row.seq_id,
            row.tenant_id,
            row.knowledge_base_id,
            row.knowledge_id,
            row.content,
            row.chunk_index,
            1 if row.is_enabled else 0,
            row.start_at,
            row.end_at,
            row.pre_chunk_id,
            row.next_chunk_id,
            row.chunk_type,
            row.parent_chunk_id,
            row.image_info,
            row.video_info,
            self._json_text(row.relation_chunks),
            self._json_text(row.indirect_relation_chunks),
            self._json_text(row.metadata),
            row.tag_id,
            row.status,
            row.content_hash,
            row.flags,
            row.created_at,
            row.updated_at,
            row.deleted_at,
        )

    def _from_row(self, row: sqlite3.Row) -> ChunkRow:
        return ChunkRow(
            id=row["id"],
            seq_id=row["seq_id"],
            tenant_id=row["tenant_id"],
            knowledge_base_id=row["knowledge_base_id"],
            knowledge_id=row["knowledge_id"],
            content=row["content"],
            chunk_index=row["chunk_index"],
            is_enabled=bool(row["is_enabled"]),
            start_at=row["start_at"],
            end_at=row["end_at"],
            pre_chunk_id=row["pre_chunk_id"] or "",
            next_chunk_id=row["next_chunk_id"] or "",
            chunk_type=row["chunk_type"],
            parent_chunk_id=row["parent_chunk_id"] or "",
            image_info=row["image_info"] or "",
            video_info=row["video_info"] or "",
            relation_chunks=row["relation_chunks"],
            indirect_relation_chunks=row["indirect_relation_chunks"],
            metadata=row["metadata"],
            tag_id=row["tag_id"] or "",
            status=row["status"],
            content_hash=row["content_hash"] or "",
            flags=row["flags"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            deleted_at=row["deleted_at"],
        )

    def _json_text(self, value):
        if value is None or isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
