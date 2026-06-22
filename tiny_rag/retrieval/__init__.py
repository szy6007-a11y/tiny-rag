"""Retrieval pipeline services ported from WeKnora's post-search chain."""

from .hybrid import (
    EngineAwareNormalizer,
    HybridSearchService,
    deduplicate_by_score,
    fuse_or_deduplicate,
    fuse_with_rrf,
)
from .merge import MergeService
from .models import (
    CHUNK_TYPE_FAQ,
    CHUNK_TYPE_PARENT_TEXT,
    CHUNK_TYPE_TEXT,
    EmbeddingModelResolver,
    KNOWLEDGE_BASE_TYPE_FAQ,
    MATCH_TYPE_DIRECT_LOAD,
    MATCH_TYPE_HISTORY,
    KnowledgeBaseRef,
    PipelineResult,
    RankResult,
    ResolvedEmbeddingModel,
    RetrievalConfig,
    SearchResult,
)
from .pipeline import SearchPipeline, retrieve_pipeline
from .rerank import RerankConfig, RerankService, SiliconFlowReranker

__all__ = [
    "CHUNK_TYPE_FAQ",
    "CHUNK_TYPE_PARENT_TEXT",
    "CHUNK_TYPE_TEXT",
    "EmbeddingModelResolver",
    "EngineAwareNormalizer",
    "HybridSearchService",
    "KNOWLEDGE_BASE_TYPE_FAQ",
    "KnowledgeBaseRef",
    "MATCH_TYPE_DIRECT_LOAD",
    "MATCH_TYPE_HISTORY",
    "MergeService",
    "PipelineResult",
    "RankResult",
    "ResolvedEmbeddingModel",
    "RerankConfig",
    "RerankService",
    "RetrievalConfig",
    "SearchPipeline",
    "SearchResult",
    "SiliconFlowReranker",
    "deduplicate_by_score",
    "fuse_or_deduplicate",
    "fuse_with_rrf",
    "retrieve_pipeline",
]
