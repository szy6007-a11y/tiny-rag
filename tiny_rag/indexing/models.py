from __future__ import annotations

from dataclasses import dataclass
from typing import Any


KEYWORDS_RETRIEVER_TYPE = "keywords"
VECTOR_RETRIEVER_TYPE = "vector"
SQLITE_RETRIEVER_ENGINE_TYPE = "sqlite"
CHUNK_SOURCE_TYPE = 0
MATCH_TYPE_EMBEDDING = 0
MATCH_TYPE_KEYWORDS = 1
DEFAULT_RETRIEVER_TYPES = (KEYWORDS_RETRIEVER_TYPE, VECTOR_RETRIEVER_TYPE)


@dataclass(frozen=True)
class IndexInfo:
    content: str
    source_id: str
    source_type: int = CHUNK_SOURCE_TYPE
    chunk_id: str = ""
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    id: str = ""
    knowledge_type: str = ""
    tag_id: str = ""
    is_enabled: bool = True
    is_recommended: bool = False


@dataclass
class BatchSaveStats:
    metadata_count: int = 0
    fts_count: int = 0
    vector_count: int = 0
    skipped_duplicate_count: int = 0
    batches: int = 0

    def add(self, other: "BatchSaveStats") -> None:
        self.metadata_count += other.metadata_count
        self.fts_count += other.fts_count
        self.vector_count += other.vector_count
        self.skipped_duplicate_count += other.skipped_duplicate_count
        self.batches += other.batches


@dataclass
class BatchIndexStats(BatchSaveStats):
    requested_count: int = 0
    indexed_count: int = 0
    deduplicated_count: int = 0
    embedded_count: int = 0

    @classmethod
    def from_save_stats(
        cls,
        save_stats: BatchSaveStats,
        *,
        requested_count: int,
        indexed_count: int,
        deduplicated_count: int,
        embedded_count: int,
    ) -> "BatchIndexStats":
        return cls(
            metadata_count=save_stats.metadata_count,
            fts_count=save_stats.fts_count,
            vector_count=save_stats.vector_count,
            skipped_duplicate_count=save_stats.skipped_duplicate_count,
            batches=save_stats.batches,
            requested_count=requested_count,
            indexed_count=indexed_count,
            deduplicated_count=deduplicated_count,
            embedded_count=embedded_count,
        )


@dataclass
class IndexKnowledgeStats(BatchIndexStats):
    knowledge_id: str = ""
    deleted_count: int = 0
    text_chunk_count: int = 0
    estimated_storage_size: int = 0


@dataclass
class RetrieveParams:
    query: str = ""
    embedding: list[float] | None = None
    knowledge_base_ids: list[str] | None = None
    knowledge_ids: list[str] | None = None
    tag_ids: list[str] | None = None
    exclude_knowledge_ids: list[str] | None = None
    exclude_chunk_ids: list[str] | None = None
    top_k: int = 0
    threshold: float = 0.0
    knowledge_type: str = ""
    additional_params: dict[str, Any] | None = None
    retriever_type: str = ""


@dataclass
class IndexWithScore:
    id: str
    content: str
    source_id: str
    source_type: int
    chunk_id: str
    knowledge_id: str
    knowledge_base_id: str
    tag_id: str
    score: float
    match_type: int
    is_enabled: bool = False

    def get_score(self) -> float:
        return self.score


@dataclass
class RetrieveResult:
    results: list[IndexWithScore] | None = None
    retriever_engine_type: str = SQLITE_RETRIEVER_ENGINE_TYPE
    retriever_type: str = ""
    error: Exception | None = None
