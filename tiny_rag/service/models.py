from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tiny_rag.agent import AgentConfig, AgentState, Message
from tiny_rag.chunking import SplitterConfig
from tiny_rag.converting.models.document import Document
from tiny_rag.indexing.models import (
    DEFAULT_RETRIEVER_TYPES,
    IndexKnowledgeStats,
    KEYWORDS_RETRIEVER_TYPE,
    VECTOR_RETRIEVER_TYPE,
)
from tiny_rag.persistence import PersistChunksResult
from tiny_rag.retrieval import KnowledgeBaseRef, PipelineResult


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    id: str
    tenant_id: int = 0
    name: str = ""
    embedding_model_id: str = ""
    embedding_model_key: str = ""
    vector_store_id: str = ""
    retriever_engine_type: str = "sqlite"
    kb_type: str = ""
    vector_enabled: bool = True
    keyword_enabled: bool = True

    def retriever_types(self) -> tuple[str, ...]:
        types: list[str] = []
        if self.keyword_enabled:
            types.append(KEYWORDS_RETRIEVER_TYPE)
        if self.vector_enabled:
            types.append(VECTOR_RETRIEVER_TYPE)
        return tuple(types)

    def to_ref(self) -> KnowledgeBaseRef:
        return KnowledgeBaseRef(
            id=self.id,
            tenant_id=self.tenant_id,
            embedding_model_id=self.embedding_model_id,
            embedding_model_key=self.embedding_model_key,
            vector_store_id=self.vector_store_id,
            retriever_engine_type=self.retriever_engine_type,
            kb_type=self.kb_type,
            vector_enabled=self.vector_enabled,
            keyword_enabled=self.keyword_enabled,
        )


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    knowledge_base_id: str
    tenant_id: int = 0
    title: str = ""
    file_name: str = ""
    file_type: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestRequest:
    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    content: bytes | str | Path
    title: str = ""
    file_name: str = ""
    file_type: str = ""
    chunk_config: SplitterConfig | None = None
    retriever_types: Sequence[str] | None = None
    knowledge_base: KnowledgeBaseConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    knowledge: KnowledgeRecord
    knowledge_base: KnowledgeBaseConfig
    document: Document
    persist_result: PersistChunksResult
    index_stats: IndexKnowledgeStats


@dataclass(frozen=True)
class QueryRequest:
    query: str
    tenant_id: int = 0
    knowledge_base_ids: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    top_k: int = 10
    vector_threshold: float = 0.6
    keyword_threshold: float = 0.5
    rerank_threshold: float = 0.3


@dataclass
class QueryResult:
    request: QueryRequest
    pipeline_result: PipelineResult


@dataclass(frozen=True)
class AgentQARequest:
    query: str
    session_id: str = ""
    tenant_id: int = 0
    agent_config: AgentConfig | None = None
    knowledge_base_ids: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class AgentQAResult:
    request: AgentQARequest
    state: AgentState


def default_knowledge_base_config(
    knowledge_base_id: str,
    *,
    tenant_id: int = 0,
    embedding_model_id: str = "",
) -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(
        id=knowledge_base_id,
        tenant_id=tenant_id,
        embedding_model_id=embedding_model_id,
        keyword_enabled=KEYWORDS_RETRIEVER_TYPE in DEFAULT_RETRIEVER_TYPES,
        vector_enabled=VECTOR_RETRIEVER_TYPE in DEFAULT_RETRIEVER_TYPES,
    )
