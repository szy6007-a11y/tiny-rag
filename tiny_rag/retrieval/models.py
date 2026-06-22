from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from tiny_rag.cancellation import CancellationToken
from tiny_rag.indexing.models import (
    MATCH_TYPE_EMBEDDING,
    MATCH_TYPE_KEYWORDS,
    SQLITE_RETRIEVER_ENGINE_TYPE,
)


MATCH_TYPE_NEAR_BY_CHUNK = 2
MATCH_TYPE_HISTORY = 3
MATCH_TYPE_PARENT_CHUNK = 4
MATCH_TYPE_RELATION_CHUNK = 5
MATCH_TYPE_GRAPH = 6
MATCH_TYPE_DIRECT_LOAD = 7
MATCH_TYPE_DATA_ANALYSIS = 8

CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_PARENT_TEXT = "parent_text"
CHUNK_TYPE_IMAGE_OCR = "image_ocr"
CHUNK_TYPE_IMAGE_CAPTION = "image_caption"
CHUNK_TYPE_SUMMARY = "summary"
CHUNK_TYPE_FAQ = "faq"
CHUNK_TYPE_TABLE_SUMMARY = "table_summary"
CHUNK_TYPE_TABLE_COLUMN = "table_column"

KNOWLEDGE_BASE_TYPE_FAQ = "faq"
KNOWLEDGE_TYPE_FAQ = "faq"


@dataclass(frozen=True)
class KnowledgeBaseRef:
    id: str
    tenant_id: int = 0
    embedding_model_id: str = ""
    embedding_model_key: str = ""
    vector_store_id: str = ""
    retriever_engine_type: str = SQLITE_RETRIEVER_ENGINE_TYPE
    kb_type: str = ""
    vector_enabled: bool = True
    keyword_enabled: bool = True

    def model_identity(self) -> str:
        return self.embedding_model_key or self.embedding_model_id

    def is_vector_enabled(self) -> bool:
        return self.vector_enabled

    def is_keyword_enabled(self) -> bool:
        return self.keyword_enabled


@dataclass
class RetrievalConfig:
    rrf_k: int = 0
    rrf_vector_weight: float = 0.0
    rrf_keyword_weight: float = 0.0

    def effective_rrf_k(self) -> int:
        return self.rrf_k if self.rrf_k > 0 else 60

    def effective_rrf_weights(self) -> tuple[float, float]:
        if self.rrf_vector_weight == 0 and self.rrf_keyword_weight == 0:
            return 0.7, 0.3
        vector = self.rrf_vector_weight if self.rrf_vector_weight > 0 else 0.7
        keyword = self.rrf_keyword_weight if self.rrf_keyword_weight > 0 else 0.3
        return vector, keyword


@dataclass
class SearchResult:
    id: str
    content: str
    knowledge_id: str = ""
    chunk_index: int = 0
    knowledge_title: str = ""
    start_at: int = 0
    end_at: int = 0
    seq: int = 0
    score: float = 0.0
    match_type: int = MATCH_TYPE_EMBEDDING
    sub_chunk_id: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    chunk_type: str = CHUNK_TYPE_TEXT
    parent_chunk_id: str = ""
    image_info: str = ""
    knowledge_filename: str = ""
    knowledge_source: str = ""
    knowledge_channel: str = ""
    chunk_metadata: Any = None
    matched_content: str = ""
    knowledge_description: str = ""
    knowledge_base_id: str = ""
    tag_id: str = ""
    source_id: str = ""
    source_type: int = 0

    def clone(self) -> "SearchResult":
        return SearchResult(
            id=self.id,
            content=self.content,
            knowledge_id=self.knowledge_id,
            chunk_index=self.chunk_index,
            knowledge_title=self.knowledge_title,
            start_at=self.start_at,
            end_at=self.end_at,
            seq=self.seq,
            score=self.score,
            match_type=self.match_type,
            sub_chunk_id=list(self.sub_chunk_id),
            metadata=dict(self.metadata or {}),
            chunk_type=self.chunk_type,
            parent_chunk_id=self.parent_chunk_id,
            image_info=self.image_info,
            knowledge_filename=self.knowledge_filename,
            knowledge_source=self.knowledge_source,
            knowledge_channel=self.knowledge_channel,
            chunk_metadata=self.chunk_metadata,
            matched_content=self.matched_content,
            knowledge_description=self.knowledge_description,
            knowledge_base_id=self.knowledge_base_id,
            tag_id=self.tag_id,
            source_id=self.source_id,
            source_type=self.source_type,
        )


@dataclass(frozen=True)
class RankResult:
    index: int
    relevance_score: float


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[RankResult]:
        ...


class EmbedderLike(Protocol):
    def embed(
        self,
        text: str,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[float]:
        ...

    def batch_embed(
        self,
        texts: list[str],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[list[float]]:
        ...

    def get_model_name(self) -> str:
        ...

    def get_dimensions(self) -> int:
        ...

    def get_model_id(self) -> str:
        ...


@dataclass(frozen=True)
class ResolvedEmbeddingModel:
    identity_key: str
    embedder: EmbedderLike | None = None


class EmbeddingModelResolver(Protocol):
    def resolve_embedding_model(self, kb: KnowledgeBaseRef) -> ResolvedEmbeddingModel:
        ...


@dataclass
class PipelineResult:
    search_results: list[SearchResult]
    rerank_results: list[SearchResult]
    merge_results: list[SearchResult]
    results: list[SearchResult]
