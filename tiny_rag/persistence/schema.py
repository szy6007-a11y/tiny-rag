from __future__ import annotations

import sqlite3


CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    knowledge_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT 1,
    start_at INTEGER NOT NULL,
    end_at INTEGER NOT NULL,
    pre_chunk_id TEXT,
    next_chunk_id TEXT,
    chunk_type TEXT NOT NULL DEFAULT 'text',
    parent_chunk_id TEXT,
    image_info TEXT,
    video_info TEXT,
    relation_chunks TEXT,
    indirect_relation_chunks TEXT,
    metadata TEXT,
    tag_id TEXT,
    status INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT,
    flags INTEGER NOT NULL DEFAULT 1,
    seq_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted_at DATETIME
);
"""


CHUNKS_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_tenant_kg ON chunks(tenant_id, knowledge_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks(chunk_type);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_tag ON chunks(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_seq_id ON chunks(seq_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_kb_tenant ON chunks(knowledge_base_id, tenant_id);",
    (
        "CREATE INDEX IF NOT EXISTS idx_chunks_knowledge_enabled "
        "ON chunks(knowledge_id, is_enabled, deleted_at);"
    ),
]


def ensure_chunks_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CHUNKS_TABLE_SQL)
    for sql in CHUNKS_INDEX_SQL:
        conn.execute(sql)
