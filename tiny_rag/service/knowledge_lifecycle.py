from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from typing import Any, Mapping, Sequence

from tiny_rag.persistence.chunk_models import utc_timestamp
from tiny_rag.persistence.sqlite import transaction

from .models import KnowledgeRecord


PARSE_STATUS_PENDING = "pending"
PARSE_STATUS_PROCESSING = "processing"
PARSE_STATUS_FINALIZING = "finalizing"
PARSE_STATUS_COMPLETED = "completed"
PARSE_STATUS_FAILED = "failed"
PARSE_STATUS_DELETING = "deleting"
PARSE_STATUS_CANCELLED = "cancelled"

ENABLE_STATUS_ENABLED = "enabled"
ENABLE_STATUS_DISABLED = "disabled"

KNOWLEDGE_TYPE_FILE = "file"
KNOWLEDGE_TYPE_MANUAL = "manual"
KNOWLEDGE_TYPE_URL = "url"


KNOWLEDGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS knowledges (
    id TEXT PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'file',
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL DEFAULT 'web',
    parse_status TEXT NOT NULL DEFAULT 'pending',
    enable_status TEXT NOT NULL DEFAULT 'disabled',
    embedding_model_id TEXT NOT NULL DEFAULT '',
    file_name TEXT NOT NULL DEFAULT '',
    file_type TEXT NOT NULL DEFAULT '',
    file_size INTEGER NOT NULL DEFAULT 0,
    file_hash TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL DEFAULT '',
    storage_size INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME,
    error_message TEXT NOT NULL DEFAULT '',
    deleted_at DATETIME
);
"""


KNOWLEDGES_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_knowledges_tenant_kb ON knowledges(tenant_id, knowledge_base_id);",
    "CREATE INDEX IF NOT EXISTS idx_knowledges_status ON knowledges(parse_status);",
    "CREATE INDEX IF NOT EXISTS idx_knowledges_file_hash ON knowledges(file_hash);",
    "CREATE INDEX IF NOT EXISTS idx_knowledges_source ON knowledges(source);",
    "CREATE INDEX IF NOT EXISTS idx_knowledges_deleted ON knowledges(deleted_at);",
]


KNOWLEDGE_COLUMNS = [
    "id",
    "tenant_id",
    "knowledge_base_id",
    "type",
    "title",
    "description",
    "source",
    "channel",
    "parse_status",
    "enable_status",
    "embedding_model_id",
    "file_name",
    "file_type",
    "file_size",
    "file_hash",
    "file_path",
    "storage_size",
    "metadata",
    "created_at",
    "updated_at",
    "processed_at",
    "error_message",
    "deleted_at",
]


UPDATE_COLUMNS = [
    "tenant_id",
    "knowledge_base_id",
    "type",
    "title",
    "description",
    "source",
    "channel",
    "parse_status",
    "enable_status",
    "embedding_model_id",
    "file_name",
    "file_type",
    "file_size",
    "file_hash",
    "file_path",
    "storage_size",
    "metadata",
    "processed_at",
    "error_message",
    "deleted_at",
]


ACTIVE_ROW_FILTER = "(deleted_at IS NULL OR deleted_at = '')"


def ensure_knowledge_schema(conn: sqlite3.Connection) -> None:
    conn.execute(KNOWLEDGES_TABLE_SQL)
    for sql in KNOWLEDGES_INDEX_SQL:
        conn.execute(sql)


class KnowledgeRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        ensure_knowledge_schema(conn)

    def create(self, record: KnowledgeRecord) -> KnowledgeRecord:
        now = utc_timestamp()
        record = _with_defaults(record, created_at=record.created_at or now, updated_at=now)
        columns = ", ".join(KNOWLEDGE_COLUMNS)
        placeholders = ", ".join("?" for _ in KNOWLEDGE_COLUMNS)
        self.conn.execute(
            f"INSERT INTO knowledges ({columns}) VALUES ({placeholders})",
            _record_values(record),
        )
        return record

    def update(self, record: KnowledgeRecord) -> KnowledgeRecord:
        now = utc_timestamp()
        record = _with_defaults(record, updated_at=now)
        assignments = ", ".join(f"{column} = ?" for column in UPDATE_COLUMNS)
        values = [_record_value(record, column) for column in UPDATE_COLUMNS]
        values.append(record.id)
        self.conn.execute(
            f"UPDATE knowledges SET {assignments} WHERE id = ?",
            values,
        )
        return record

    def upsert(self, record: KnowledgeRecord) -> KnowledgeRecord:
        existing = self.get_by_id(record.id, include_deleted=True)
        if existing is None:
            return self.create(record)
        if existing.created_at and not record.created_at:
            record = replace(record, created_at=existing.created_at)
        return self.update(record)

    def update_columns(self, knowledge_id: str, values: Mapping[str, Any]) -> None:
        allowed = set(UPDATE_COLUMNS)
        pairs = [(key, value) for key, value in values.items() if key in allowed]
        if not pairs:
            return
        if "updated_at" not in values:
            pairs.append(("updated_at", utc_timestamp()))
        assignments = ", ".join(f"{key} = ?" for key, _ in pairs)
        self.conn.execute(
            f"UPDATE knowledges SET {assignments} WHERE id = ?",
            [self._encode_column_value(key, value) for key, value in pairs] + [knowledge_id],
        )

    def mark_status(
        self,
        knowledge_id: str,
        status: str,
        *,
        error_message: str = "",
        enable_status: str | None = None,
        processed_at: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"parse_status": status}
        if error_message:
            values["error_message"] = error_message
        if enable_status is not None:
            values["enable_status"] = enable_status
        if processed_at is not None:
            values["processed_at"] = processed_at
        self.update_columns(knowledge_id, values)

    def get(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        include_deleted: bool = False,
    ) -> KnowledgeRecord | None:
        deleted_filter = "" if include_deleted else f"AND {ACTIVE_ROW_FILTER}"
        row = self.conn.execute(
            f"""
            SELECT {", ".join(KNOWLEDGE_COLUMNS)}
            FROM knowledges
            WHERE tenant_id = ?
              AND id = ?
              {deleted_filter}
            """,
            (tenant_id, knowledge_id),
        ).fetchone()
        return _record_from_row(row) if row is not None else None

    def get_by_id(self, knowledge_id: str, *, include_deleted: bool = False) -> KnowledgeRecord | None:
        deleted_filter = "" if include_deleted else f"AND {ACTIVE_ROW_FILTER}"
        row = self.conn.execute(
            f"""
            SELECT {", ".join(KNOWLEDGE_COLUMNS)}
            FROM knowledges
            WHERE id = ?
              {deleted_filter}
            """,
            (knowledge_id,),
        ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list_by_knowledge_base(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        include_deleted: bool = False,
    ) -> list[KnowledgeRecord]:
        deleted_filter = "" if include_deleted else f"AND {ACTIVE_ROW_FILTER}"
        rows = self.conn.execute(
            f"""
            SELECT {", ".join(KNOWLEDGE_COLUMNS)}
            FROM knowledges
            WHERE tenant_id = ?
              AND knowledge_base_id = ?
              {deleted_filter}
            ORDER BY created_at ASC, id ASC
            """,
            (tenant_id, knowledge_base_id),
        ).fetchall()
        return [_record_from_row(row) for row in rows]

    def check_exists(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_type: str,
        file_hash: str = "",
        file_name: str = "",
        file_size: int = 0,
        source: str = "",
    ) -> KnowledgeRecord | None:
        base_params: list[Any] = [tenant_id, knowledge_base_id, PARSE_STATUS_FAILED]
        base_where = f"""
            tenant_id = ?
            AND knowledge_base_id = ?
            AND parse_status <> ?
            AND {ACTIVE_ROW_FILTER}
        """

        if source:
            row = self.conn.execute(
                f"""
                SELECT {", ".join(KNOWLEDGE_COLUMNS)}
                FROM knowledges
                WHERE {base_where}
                  AND source = ?
                LIMIT 1
                """,
                (*base_params, source),
            ).fetchone()
            if row is not None:
                return _record_from_row(row)

        if file_hash:
            row = self.conn.execute(
                f"""
                SELECT {", ".join(KNOWLEDGE_COLUMNS)}
                FROM knowledges
                WHERE {base_where}
                  AND file_hash = ?
                  AND type = ?
                LIMIT 1
                """,
                (*base_params, file_hash, knowledge_type),
            ).fetchone()
            if row is not None:
                return _record_from_row(row)

        if file_name and file_size > 0:
            row = self.conn.execute(
                f"""
                SELECT {", ".join(KNOWLEDGE_COLUMNS)}
                FROM knowledges
                WHERE {base_where}
                  AND file_name = ?
                  AND file_size = ?
                  AND type = ?
                LIMIT 1
                """,
                (*base_params, file_name, file_size, knowledge_type),
            ).fetchone()
            if row is not None:
                return _record_from_row(row)

        return None

    def soft_delete(self, *, tenant_id: int, knowledge_id: str) -> None:
        now = utc_timestamp()
        self.conn.execute(
            f"""
            UPDATE knowledges
            SET deleted_at = ?,
                updated_at = ?,
                parse_status = ?
            WHERE tenant_id = ?
              AND id = ?
              AND {ACTIVE_ROW_FILTER}
            """,
            (now, now, PARSE_STATUS_DELETING, tenant_id, knowledge_id),
        )

    def active_records_map(self, *, tenant_id: int, knowledge_base_id: str) -> dict[str, KnowledgeRecord]:
        return {
            record.id: record
            for record in self.list_by_knowledge_base(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )
        }

    def _encode_column_value(self, key: str, value: Any) -> Any:
        if key == "metadata":
            return _metadata_to_json(value)
        if key in {"processed_at", "deleted_at"} and not value:
            return None
        return value


def _with_defaults(
    record: KnowledgeRecord,
    *,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> KnowledgeRecord:
    return replace(
        record,
        type=record.type or KNOWLEDGE_TYPE_FILE,
        title=record.title or record.file_name or record.id,
        channel=record.channel or "web",
        parse_status=record.parse_status or PARSE_STATUS_PENDING,
        enable_status=record.enable_status or ENABLE_STATUS_DISABLED,
        embedding_model_id=record.embedding_model_id or "",
        metadata=dict(record.metadata or {}),
        created_at=created_at if created_at is not None else record.created_at,
        updated_at=updated_at if updated_at is not None else record.updated_at,
    )


def _record_values(record: KnowledgeRecord) -> tuple[Any, ...]:
    return tuple(_record_value(record, column) for column in KNOWLEDGE_COLUMNS)


def _record_value(record: KnowledgeRecord, column: str) -> Any:
    if column == "metadata":
        return _metadata_to_json(record.metadata)
    if column in {"processed_at", "deleted_at"} and not getattr(record, column):
        return None
    return getattr(record, column)


def _metadata_to_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _metadata_from_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _record_from_row(row: sqlite3.Row) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=str(row["id"]),
        tenant_id=int(row["tenant_id"]),
        knowledge_base_id=str(row["knowledge_base_id"]),
        type=str(row["type"] or KNOWLEDGE_TYPE_FILE),
        title=str(row["title"] or ""),
        description=str(row["description"] or ""),
        source=str(row["source"] or ""),
        channel=str(row["channel"] or "web"),
        parse_status=str(row["parse_status"] or PARSE_STATUS_PENDING),
        enable_status=str(row["enable_status"] or ENABLE_STATUS_DISABLED),
        embedding_model_id=str(row["embedding_model_id"] or ""),
        file_name=str(row["file_name"] or ""),
        file_type=str(row["file_type"] or ""),
        file_size=int(row["file_size"] or 0),
        file_hash=str(row["file_hash"] or ""),
        file_path=str(row["file_path"] or ""),
        storage_size=int(row["storage_size"] or 0),
        metadata=_metadata_from_json(row["metadata"]),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
        processed_at=str(row["processed_at"] or ""),
        error_message=str(row["error_message"] or ""),
        deleted_at=str(row["deleted_at"] or ""),
    )


def replace_record(record: KnowledgeRecord, **changes: Any) -> KnowledgeRecord:
    return replace(record, **changes)
