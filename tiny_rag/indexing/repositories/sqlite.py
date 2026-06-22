from __future__ import annotations

from pathlib import Path
import logging
import sqlite3
import threading
import unicodedata
from uuid import uuid4
from typing import Callable, Mapping, Sequence, TypeVar

import sqlite_vec

from tiny_rag.indexing.models import (
    BatchSaveStats,
    IndexInfo,
    IndexWithScore,
    KEYWORDS_RETRIEVER_TYPE,
    MATCH_TYPE_EMBEDDING,
    MATCH_TYPE_KEYWORDS,
    RetrieveParams,
    RetrieveResult,
    SQLITE_RETRIEVER_ENGINE_TYPE,
    VECTOR_RETRIEVER_TYPE,
)


T = TypeVar("T")


LITE_EMBEDDINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS lite_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    source_id TEXT NOT NULL,
    source_type INTEGER NOT NULL,
    chunk_id TEXT,
    knowledge_id TEXT,
    knowledge_base_id TEXT,
    tag_id TEXT,
    content TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    is_enabled BOOLEAN DEFAULT 1,
    UNIQUE(source_id, source_type)
);
"""


LITE_EMBEDDINGS_INDEX_SQL = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sqlite_emb_source "
    "ON lite_embeddings(source_id, source_type);",
    "CREATE INDEX IF NOT EXISTS idx_sqlite_emb_chunk_id ON lite_embeddings(chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_sqlite_emb_knowledge_id ON lite_embeddings(knowledge_id);",
    "CREATE INDEX IF NOT EXISTS idx_sqlite_emb_kb_id ON lite_embeddings(knowledge_base_id);",
    "CREATE INDEX IF NOT EXISTS idx_sqlite_emb_tag_id ON lite_embeddings(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_sqlite_emb_enabled ON lite_embeddings(is_enabled);",
]


FTS5_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS lite_embeddings_fts USING fts5(
    content, source_id, chunk_id, knowledge_id, knowledge_base_id,
    content='',
    contentless_delete=1,
    tokenize='unicode61'
);
"""


def connect_sqlite_index_db(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def vec_table_name(dim: int) -> str:
    return f"vec_embeddings_{int(dim)}"


def clean_invalid_utf8(value: str) -> str:
    return value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").replace(
        "\x00", ""
    )


def tokenize_cjk_bigram(text: str) -> str:
    parts: list[str] = []
    current_cjk: list[str] = []
    current_non_cjk: list[str] = []

    def flush_cjk() -> None:
        if not current_cjk:
            return
        if len(current_cjk) == 1:
            parts.append(current_cjk[0])
        else:
            for index in range(len(current_cjk) - 1):
                parts.append(current_cjk[index] + current_cjk[index + 1])
        current_cjk.clear()

    def flush_non_cjk() -> None:
        if current_non_cjk:
            parts.append("".join(current_non_cjk))
            current_non_cjk.clear()

    for char in text:
        if _is_han(char):
            flush_non_cjk()
            current_cjk.append(char)
        elif _is_delimiter(char):
            flush_cjk()
            flush_non_cjk()
        else:
            flush_cjk()
            current_non_cjk.append(char)

    flush_cjk()
    flush_non_cjk()
    return " ".join(parts)


def _is_han(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0x2CEB0 <= code <= 0x2EBEF
        or 0x30000 <= code <= 0x3134F
        or 0x31350 <= code <= 0x323AF
    )


def _is_delimiter(char: str) -> bool:
    if char.isspace():
        return True
    category = unicodedata.category(char)
    return category.startswith("P") or category.startswith("S")


class SQLiteIndexRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._vec_tables: set[int] = set()
        self._load_vec()
        self._ensure_schema()

    def engine_type(self) -> str:
        return SQLITE_RETRIEVER_ENGINE_TYPE

    def support(self) -> list[str]:
        return [KEYWORDS_RETRIEVER_TYPE, VECTOR_RETRIEVER_TYPE]

    def EngineType(self) -> str:
        return self.engine_type()

    def Support(self) -> list[str]:
        return self.support()

    def retrieve(self, params: RetrieveParams) -> list[RetrieveResult]:
        results: list[RetrieveResult] = []

        if params.retriever_type in {KEYWORDS_RETRIEVER_TYPE, ""}:
            try:
                results.extend(self.keywords_retrieve(params))
            except Exception as exc:
                results.append(
                    RetrieveResult(
                        retriever_engine_type=SQLITE_RETRIEVER_ENGINE_TYPE,
                        retriever_type=KEYWORDS_RETRIEVER_TYPE,
                        error=exc,
                    )
                )

        if params.retriever_type in {VECTOR_RETRIEVER_TYPE, ""}:
            try:
                results.extend(self.vector_retrieve(params))
            except Exception as exc:
                results.append(
                    RetrieveResult(
                        retriever_engine_type=SQLITE_RETRIEVER_ENGINE_TYPE,
                        retriever_type=VECTOR_RETRIEVER_TYPE,
                        error=exc,
                    )
                )

        return results

    def save(
        self,
        index_info: IndexInfo,
        *,
        embeddings_by_source_id: Mapping[str, Sequence[float]] | None = None,
    ) -> BatchSaveStats:
        return self.batch_save(
            [index_info],
            embeddings_by_source_id=embeddings_by_source_id,
        )

    def batch_save(
        self,
        index_info_list: Sequence[IndexInfo],
        *,
        embeddings_by_source_id: Mapping[str, Sequence[float]] | None = None,
    ) -> BatchSaveStats:
        infos = list(index_info_list)
        if not infos:
            return BatchSaveStats()

        embeddings = embeddings_by_source_id or {}
        stats = BatchSaveStats(batches=1)

        def write() -> BatchSaveStats:
            for info in infos:
                embedding = embeddings.get(info.source_id)
                has_embedding = embedding is not None and len(embedding) > 0
                dimension = len(embedding) if has_embedding else 0
                cursor = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO lite_embeddings (
                        source_id, source_type, chunk_id, knowledge_id,
                        knowledge_base_id, tag_id, content, dimension,
                        is_enabled, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        info.source_id,
                        int(info.source_type),
                        info.chunk_id,
                        info.knowledge_id,
                        info.knowledge_base_id,
                        info.tag_id,
                        clean_invalid_utf8(info.content),
                        dimension,
                        1 if info.is_enabled else 0,
                    ),
                )
                if cursor.rowcount == 0:
                    stats.skipped_duplicate_count += 1
                    continue

                row_id = int(cursor.lastrowid)
                stats.metadata_count += 1
                self._sync_fts5_insert(
                    row_id=row_id,
                    content=info.content,
                    source_id=info.source_id,
                    chunk_id=info.chunk_id,
                    knowledge_id=info.knowledge_id,
                    knowledge_base_id=info.knowledge_base_id,
                )
                stats.fts_count += 1

                if has_embedding:
                    self._insert_vec(row_id, dimension, embedding)
                    stats.vector_count += 1
            return stats

        return self._write_transaction(write)

    def delete_by_chunk_id_list(
        self,
        chunk_id_list: Sequence[str],
        *,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        del dimension, knowledge_type
        return self._delete_by_column("chunk_id", chunk_id_list)

    def delete_by_source_id_list(
        self,
        source_id_list: Sequence[str],
        *,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        del dimension, knowledge_type
        return self._delete_by_column("source_id", source_id_list)

    def delete_by_knowledge_id_list(
        self,
        knowledge_id_list: Sequence[str],
        *,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        del dimension, knowledge_type
        return self._delete_by_column("knowledge_id", knowledge_id_list)

    def estimate_storage_size(
        self,
        index_info_list: Sequence[IndexInfo],
        *,
        embeddings_by_source_id: Mapping[str, Sequence[float]] | None = None,
    ) -> int:
        del embeddings_by_source_id
        return sum(len(info.content.encode("utf-8")) + 200 for info in index_info_list)

    def copy_indices(
        self,
        source_knowledge_base_id: str,
        source_to_target_kb_id_map: Mapping[str, str],
        source_to_target_chunk_id_map: Mapping[str, str],
        target_knowledge_base_id: str,
        *,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> None:
        del source_knowledge_base_id, dimension, knowledge_type

        def write() -> None:
            for source_chunk_id, target_chunk_id in source_to_target_chunk_id_map.items():
                src = self.conn.execute(
                    """
                    SELECT id, source_type, chunk_id, knowledge_id, knowledge_base_id,
                           tag_id, content, dimension, is_enabled
                    FROM lite_embeddings
                    WHERE chunk_id = ?
                    ORDER BY id
                    LIMIT 1
                    """,
                    (source_chunk_id,),
                ).fetchone()
                if src is None:
                    continue

                new_source_id = str(uuid4())
                target_knowledge_id = source_to_target_kb_id_map.get(src["knowledge_id"], "")
                try:
                    cursor = self.conn.execute(
                        """
                        INSERT INTO lite_embeddings (
                            source_id, source_type, chunk_id, knowledge_id,
                            knowledge_base_id, tag_id, content, dimension, is_enabled,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (
                            new_source_id,
                            src["source_type"],
                            target_chunk_id,
                            target_knowledge_id,
                            target_knowledge_base_id,
                            src["tag_id"],
                            src["content"],
                            src["dimension"],
                            src["is_enabled"],
                        ),
                    )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "[SQLite] CopyIndices: failed to copy chunk %s: %s",
                        source_chunk_id,
                        exc,
                    )
                    continue
                new_row_id = int(cursor.lastrowid)
                self._sync_fts5_insert(
                    row_id=new_row_id,
                    content=src["content"],
                    source_id=new_source_id,
                    chunk_id=target_chunk_id,
                    knowledge_id=target_knowledge_id,
                    knowledge_base_id=target_knowledge_base_id,
                )
                if int(src["dimension"] or 0) > 0 and new_row_id > 0:
                    self._copy_vec(int(src["id"]), new_row_id, int(src["dimension"]))

        self._write_transaction(write)

    def batch_update_chunk_enabled_status(self, chunk_status_map: Mapping[str, bool]) -> None:
        def write() -> None:
            for chunk_id, enabled in chunk_status_map.items():
                self.conn.execute(
                    "UPDATE lite_embeddings SET is_enabled = ?, updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?",
                    (1 if enabled else 0, chunk_id),
                )

        self._write_transaction(write)

    def batch_update_chunk_tag_id(self, chunk_tag_map: Mapping[str, str]) -> None:
        def write() -> None:
            for chunk_id, tag_id in chunk_tag_map.items():
                self.conn.execute(
                    "UPDATE lite_embeddings SET tag_id = ?, updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?",
                    (tag_id, chunk_id),
                )

        self._write_transaction(write)

    def keywords_retrieve(self, params: RetrieveParams) -> list[RetrieveResult]:
        if params.query == "":
            return []

        fts_query = sanitize_fts5_query(params.query)
        sql = """
            SELECT e.id, e.source_id, e.source_type, e.chunk_id,
                e.knowledge_id, e.knowledge_base_id, e.tag_id,
                e.content,
                (bm25(lite_embeddings_fts) * -1000000.0) AS score
            FROM lite_embeddings_fts
            JOIN lite_embeddings e ON e.id = lite_embeddings_fts.rowid
            WHERE lite_embeddings_fts MATCH ?
            AND (e.is_enabled IS NULL OR e.is_enabled = 1)
        """
        args: list[object] = [fts_query]

        for clause, clause_args in _build_filter_where(params):
            sql += " AND " + clause
            args.extend(clause_args)

        sql += " ORDER BY score DESC LIMIT ?"
        args.append(params.top_k)

        rows = self.conn.execute(sql, args).fetchall()
        items = [
            IndexWithScore(
                id=str(row["id"]),
                source_id=row["source_id"],
                source_type=int(row["source_type"]),
                chunk_id=row["chunk_id"],
                knowledge_id=row["knowledge_id"],
                knowledge_base_id=row["knowledge_base_id"],
                tag_id=row["tag_id"] or "",
                content=row["content"],
                score=float(row["score"]),
                match_type=MATCH_TYPE_KEYWORDS,
            )
            for row in rows
        ]
        return [
            RetrieveResult(
                results=items,
                retriever_engine_type=SQLITE_RETRIEVER_ENGINE_TYPE,
                retriever_type=KEYWORDS_RETRIEVER_TYPE,
            )
        ]

    def vector_retrieve(self, params: RetrieveParams) -> list[RetrieveResult]:
        if not params.embedding:
            return []

        dim = len(params.embedding)
        self._ensure_vec_table(dim)
        query_blob = sqlite_vec.serialize_float32([float(value) for value in params.embedding])
        table_name = vec_table_name(dim)
        sql = f"""
            SELECT v.rowid, v.distance,
                e.source_id, e.source_type, e.chunk_id,
                e.knowledge_id, e.knowledge_base_id,
                e.tag_id, e.content
            FROM {table_name} v
            JOIN lite_embeddings e ON e.id = v.rowid
            WHERE v.embedding MATCH ?
            AND k = ?
            AND (e.is_enabled IS NULL OR e.is_enabled = 1)
        """
        args: list[object] = [query_blob, params.top_k]

        for clause, clause_args in _build_filter_where(params):
            sql += " AND " + clause
            args.extend(clause_args)

        sql += " ORDER BY v.distance ASC"

        rows = self.conn.execute(sql, args).fetchall()
        items = [
            IndexWithScore(
                id=str(row["rowid"]),
                source_id=row["source_id"],
                source_type=int(row["source_type"]),
                chunk_id=row["chunk_id"],
                knowledge_id=row["knowledge_id"],
                knowledge_base_id=row["knowledge_base_id"],
                tag_id=row["tag_id"] or "",
                content=row["content"],
                score=1 - float(row["distance"]),
                match_type=MATCH_TYPE_EMBEDDING,
            )
            for row in rows
        ]
        return [
            RetrieveResult(
                results=items,
                retriever_engine_type=SQLITE_RETRIEVER_ENGINE_TYPE,
                retriever_type=VECTOR_RETRIEVER_TYPE,
            )
        ]

    def BatchSave(
        self,
        index_info_list: Sequence[IndexInfo],
        params: Mapping[str, object] | None = None,
    ) -> BatchSaveStats:
        params = params or {}
        embeddings = params.get("embedding")
        if not isinstance(embeddings, Mapping):
            embeddings = None
        return self.batch_save(index_info_list, embeddings_by_source_id=embeddings)

    def Save(
        self,
        index_info: IndexInfo,
        params: Mapping[str, object] | None = None,
    ) -> BatchSaveStats:
        params = params or {}
        embeddings = params.get("embedding")
        if not isinstance(embeddings, Mapping):
            embeddings = None
        return self.save(index_info, embeddings_by_source_id=embeddings)

    def Retrieve(self, params: RetrieveParams) -> list[RetrieveResult]:
        return self.retrieve(params)

    def DeleteByKnowledgeIDList(
        self,
        knowledge_id_list: Sequence[str],
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        return self.delete_by_knowledge_id_list(
            knowledge_id_list,
            dimension=dimension,
            knowledge_type=knowledge_type,
        )

    def DeleteByChunkIDList(
        self,
        chunk_id_list: Sequence[str],
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        return self.delete_by_chunk_id_list(
            chunk_id_list,
            dimension=dimension,
            knowledge_type=knowledge_type,
        )

    def DeleteBySourceIDList(
        self,
        source_id_list: Sequence[str],
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        return self.delete_by_source_id_list(
            source_id_list,
            dimension=dimension,
            knowledge_type=knowledge_type,
        )

    def EstimateStorageSize(
        self,
        index_info_list: Sequence[IndexInfo],
        params: Mapping[str, object] | None = None,
    ) -> int:
        del params
        return self.estimate_storage_size(index_info_list)

    def CopyIndices(
        self,
        source_knowledge_base_id: str,
        source_to_target_kb_id_map: Mapping[str, str],
        source_to_target_chunk_id_map: Mapping[str, str],
        target_knowledge_base_id: str,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> None:
        self.copy_indices(
            source_knowledge_base_id,
            source_to_target_kb_id_map,
            source_to_target_chunk_id_map,
            target_knowledge_base_id,
            dimension=dimension,
            knowledge_type=knowledge_type,
        )

    def BatchUpdateChunkEnabledStatus(self, chunk_status_map: Mapping[str, bool]) -> None:
        self.batch_update_chunk_enabled_status(chunk_status_map)

    def BatchUpdateChunkTagID(self, chunk_tag_map: Mapping[str, str]) -> None:
        self.batch_update_chunk_tag_id(chunk_tag_map)

    def _load_vec(self) -> None:
        self.conn.enable_load_extension(True)
        try:
            sqlite_vec.load(self.conn)
        finally:
            self.conn.enable_load_extension(False)

    def _ensure_schema(self) -> None:
        with self._lock:
            already_in_transaction = self.conn.in_transaction
            self.conn.execute(LITE_EMBEDDINGS_TABLE_SQL)
            for sql in LITE_EMBEDDINGS_INDEX_SQL:
                self.conn.execute(sql)
            self._init_fts5()
            self._ensure_existing_vec_tables()
            if not already_in_transaction:
                self.conn.commit()

    def _init_fts5(self) -> None:
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='lite_embeddings_fts'"
        ).fetchone()
        sql = row["sql"] if row is not None else ""
        if sql and "content='lite_embeddings'" in sql:
            self.conn.execute("DROP TABLE IF EXISTS lite_embeddings_fts")
            sql = ""
        if sql:
            return

        self.conn.execute(FTS5_TABLE_SQL)
        rows = self.conn.execute(
            """
            SELECT id, content, source_id, chunk_id, knowledge_id, knowledge_base_id
            FROM lite_embeddings
            """
        ).fetchall()
        for row in rows:
            self._sync_fts5_insert(
                row_id=row["id"],
                content=row["content"],
                source_id=row["source_id"],
                chunk_id=row["chunk_id"],
                knowledge_id=row["knowledge_id"],
                knowledge_base_id=row["knowledge_base_id"],
            )

    def _ensure_existing_vec_tables(self) -> None:
        rows = self.conn.execute(
            "SELECT DISTINCT dimension FROM lite_embeddings WHERE dimension > 0"
        ).fetchall()
        for row in rows:
            self._ensure_vec_table(int(row["dimension"]))

    def _ensure_vec_table(self, dim: int) -> None:
        if dim <= 0 or dim in self._vec_tables:
            return
        table_name = vec_table_name(dim)
        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}
            USING vec0(embedding float[{dim}] distance_metric=cosine)
            """
        )
        self._vec_tables.add(dim)

    def _insert_vec(self, row_id: int, dim: int, embedding: Sequence[float]) -> None:
        self._ensure_vec_table(dim)
        blob = sqlite_vec.serialize_float32([float(value) for value in embedding])
        self.conn.execute(
            f"INSERT INTO {vec_table_name(dim)}(rowid, embedding) VALUES (?, ?)",
            (row_id, blob),
        )

    def _delete_by_column(self, column: str, values: Sequence[str]) -> int:
        values_list = list(values)
        if not values_list:
            return 0
        placeholders = ", ".join("?" for _ in values_list)

        def write() -> int:
            rows = self.conn.execute(
                f"""
                SELECT id, dimension
                FROM lite_embeddings
                WHERE {column} IN ({placeholders})
                """,
                values_list,
            ).fetchall()
            self._delete_rows_and_vecs(rows)
            cursor = self.conn.execute(
                f"DELETE FROM lite_embeddings WHERE {column} IN ({placeholders})",
                values_list,
            )
            return int(cursor.rowcount)

        return self._write_transaction(write)

    def _sync_fts5_insert(
        self,
        *,
        row_id: int,
        content: str,
        source_id: str,
        chunk_id: str,
        knowledge_id: str,
        knowledge_base_id: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO lite_embeddings_fts(
                rowid, content, source_id, chunk_id, knowledge_id, knowledge_base_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                tokenize_cjk_bigram(clean_invalid_utf8(content)),
                source_id,
                chunk_id,
                knowledge_id,
                knowledge_base_id,
            ),
        )

    def _delete_rows_and_vecs(self, rows: Sequence[sqlite3.Row]) -> None:
        for row in rows:
            row_id = int(row["id"])
            dimension = int(row["dimension"] or 0)
            if dimension > 0 and dimension in self._vec_tables:
                self.conn.execute(
                    f"DELETE FROM {vec_table_name(dimension)} WHERE rowid = ?",
                    (row_id,),
                )
            self.conn.execute("DELETE FROM lite_embeddings_fts WHERE rowid = ?", (row_id,))

    def _copy_vec(self, source_id: int, target_id: int, dim: int) -> None:
        if dim not in self._vec_tables:
            return
        table_name = vec_table_name(dim)
        self.conn.execute(
            f"""
            INSERT INTO {table_name}(rowid, embedding)
            SELECT ?, embedding FROM {table_name} WHERE rowid = ?
            """,
            (target_id, source_id),
        )

    def _write_transaction(self, fn: Callable[[], T]) -> T:
        with self._lock:
            already_in_transaction = self.conn.in_transaction
            if not already_in_transaction:
                self.conn.execute("BEGIN")
            try:
                result = fn()
            except Exception:
                if not already_in_transaction:
                    self.conn.rollback()
                raise
            if not already_in_transaction:
                self.conn.commit()
            return result


tokenizeCJKBigram = tokenize_cjk_bigram


def sanitize_fts5_query(query: str) -> str:
    query = query.strip()
    if query == "":
        return query

    tokenized = tokenize_cjk_bigram(query)
    fields = tokenized.split()
    parts = [f'"{field}"' for field in fields if field != ""]
    if not parts:
        return ""
    return " OR ".join(parts)


def _build_filter_where(params: RetrieveParams) -> list[tuple[str, list[object]]]:
    parts: list[tuple[str, list[object]]] = []
    if params.knowledge_base_ids:
        parts.append(
            (
                "e.knowledge_base_id IN (" + _placeholders(len(params.knowledge_base_ids)) + ")",
                list(params.knowledge_base_ids),
            )
        )
    if params.knowledge_ids:
        parts.append(
            (
                "e.knowledge_id IN (" + _placeholders(len(params.knowledge_ids)) + ")",
                list(params.knowledge_ids),
            )
        )
    if params.tag_ids:
        parts.append(
            (
                "e.tag_id IN (" + _placeholders(len(params.tag_ids)) + ")",
                list(params.tag_ids),
            )
        )
    return parts


def _placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))
