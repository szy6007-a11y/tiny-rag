from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from tiny_rag.chunking.types import Diagnostics


CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_PARENT_TEXT = "parent_text"
DEFAULT_FLAGS = 0
DEFAULT_STATUS = 0


def new_chunk_id() -> str:
    return str(uuid4())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ChunkRow:
    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str
    content: str
    chunk_index: int
    start_at: int
    end_at: int
    chunk_type: str = CHUNK_TYPE_TEXT
    id: str = field(default_factory=new_chunk_id)
    seq_id: Optional[int] = None
    is_enabled: bool = True
    pre_chunk_id: str = ""
    next_chunk_id: str = ""
    parent_chunk_id: str = ""
    image_info: str = ""
    video_info: str = ""
    relation_chunks: Optional[str] = None
    indirect_relation_chunks: Optional[str] = None
    metadata: Optional[str] = None
    tag_id: str = ""
    status: int = DEFAULT_STATUS
    content_hash: str = ""
    flags: int = DEFAULT_FLAGS
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    deleted_at: Optional[str] = None


@dataclass
class PersistedChunk:
    row: ChunkRow
    context_header: str = ""

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def content(self) -> str:
        return self.row.content

    @property
    def chunk_index(self) -> int:
        return self.row.chunk_index

    @property
    def seq_id(self) -> Optional[int]:
        return self.row.seq_id

    @property
    def chunk_type(self) -> str:
        return self.row.chunk_type

    @property
    def start_at(self) -> int:
        return self.row.start_at

    @property
    def end_at(self) -> int:
        return self.row.end_at

    def __getattr__(self, name: str):
        return getattr(self.row, name)


@dataclass
class PersistChunksStats:
    inserted_count: int = 0
    text_count: int = 0
    parent_count: int = 0
    skipped_empty_count: int = 0
    deleted_count: int = 0


@dataclass
class PersistChunksResult:
    inserted_chunks: list[PersistedChunk]
    diagnostics: Diagnostics
    stats: PersistChunksStats

    @property
    def inserted_count(self) -> int:
        return self.stats.inserted_count

    @property
    def text_count(self) -> int:
        return self.stats.text_count

    @property
    def parent_count(self) -> int:
        return self.stats.parent_count
