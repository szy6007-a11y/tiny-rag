from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from typing import Mapping, Sequence

from tiny_rag.agent import AgentConfig, AgentEngine, ChatModel
from tiny_rag.agent.tools import (
    GetDocumentInfoTool,
    GrepChunksTool,
    KnowledgeSearchTool,
    ListKnowledgeChunksTool,
    SearchTarget,
    SearchTargets,
    ToolGetDocumentInfo,
    ToolGrepChunks,
    ToolKnowledgeSearch,
    ToolListKnowledgeChunks,
    ToolRegistry,
    default_allowed_tools,
)
from tiny_rag.embedding import Embedder
from tiny_rag.indexing.repositories.base import IndexRepository
from tiny_rag.indexing.repositories.sqlite import SQLiteIndexRepository
from tiny_rag.persistence import ChunkRepository
from tiny_rag.retrieval import (
    HybridSearchService,
    MergeService,
    PipelineResult,
    RerankService,
    SearchPipeline,
    SiliconFlowReranker,
)

from .models import KnowledgeBaseConfig, KnowledgeRecord, QueryRequest, QueryResult


RAG_TOOL_SET = {
    ToolKnowledgeSearch,
    ToolGrepChunks,
    ToolListKnowledgeChunks,
    ToolGetDocumentInfo,
}


class AgentService:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        repository: IndexRepository | None = None,
        chunk_repository: ChunkRepository | None = None,
        embedder: Embedder | None = None,
        knowledge_bases: Mapping[str, KnowledgeBaseConfig] | None = None,
        knowledge_records: Mapping[str, KnowledgeRecord] | None = None,
        rerank_service: RerankService | None = None,
        merge_service: MergeService | None = None,
    ) -> None:
        self.conn = conn
        self.repository = repository or SQLiteIndexRepository(conn)
        self.chunk_repository = chunk_repository or ChunkRepository(conn)
        self.embedder = embedder
        self.knowledge_bases = dict(knowledge_bases or {})
        self.knowledge_records = dict(knowledge_records or {})
        self._rerank_service_provided = rerank_service is not None
        self.rerank_service = rerank_service or default_rerank_service_from_env()
        self._uses_default_merge_service = merge_service is None
        self.merge_service = merge_service or MergeService(self.chunk_repository)

    def create_search_pipeline(
        self,
        config: AgentConfig | None = None,
        *,
        tenant_id: int = 0,
        require_reranker: bool = True,
    ) -> SearchPipeline:
        if require_reranker:
            self._ensure_knowledge_search_reranker_configured()
        kb_refs = {kb_id: kb.to_ref() for kb_id, kb in self.knowledge_bases.items()}
        hybrid = HybridSearchService(
            self.repository,
            embedder=self.embedder,
            chunk_repository=self.chunk_repository,
            tenant_id=tenant_id,
            knowledge_bases=kb_refs,
        )
        merge_service = (
            MergeService(self.chunk_repository, tenant_id=tenant_id)
            if self._uses_default_merge_service
            else self.merge_service
        )
        return SearchPipeline(
            hybrid,
            self.rerank_service,
            merge_service,
            tenant_id=tenant_id,
        )

    def create_tool_registry(
        self,
        config: AgentConfig,
        *,
        tenant_id: int = 0,
    ) -> ToolRegistry:
        registry = ToolRegistry()
        registry.set_max_tool_output_size(config.max_tool_output_chars)
        targets = self.build_search_targets(config, tenant_id=tenant_id)
        allowed_tools = _dedup(config.allowed_tools or default_allowed_tools())

        has_rag_scope = bool(targets)
        for tool_name in allowed_tools:
            if tool_name in RAG_TOOL_SET and not has_rag_scope:
                continue

            tool = None
            if tool_name == ToolKnowledgeSearch:
                tool = KnowledgeSearchTool(
                    self.create_search_pipeline(
                        config,
                        tenant_id=tenant_id,
                        require_reranker=True,
                    ),
                    search_targets=targets,
                    knowledge_base_types={
                        kb_id: kb.kb_type for kb_id, kb in self.knowledge_bases.items()
                    },
                )
            elif tool_name == ToolGrepChunks:
                tool = GrepChunksTool(
                    self.chunk_repository,
                    search_targets=targets,
                    knowledge_titles={
                        knowledge_id: record.title
                        for knowledge_id, record in self.knowledge_records.items()
                    },
                )
            elif tool_name == ToolListKnowledgeChunks:
                tool = ListKnowledgeChunksTool(self.chunk_repository, search_targets=targets)
            elif tool_name == ToolGetDocumentInfo:
                tool = GetDocumentInfoTool(
                    self.chunk_repository,
                    search_targets=targets,
                    knowledge_titles={
                        knowledge_id: record.title
                        for knowledge_id, record in self.knowledge_records.items()
                    },
                )

            if tool is not None:
                registry.register_tool(tool)
        return registry

    def create_agent_engine(
        self,
        config: AgentConfig,
        chat_model: ChatModel,
        *,
        tenant_id: int = 0,
    ) -> AgentEngine:
        return AgentEngine(
            config,
            chat_model,
            self.create_tool_registry(config, tenant_id=tenant_id),
        )

    def build_search_targets(
        self,
        config: AgentConfig,
        *,
        tenant_id: int = 0,
    ) -> SearchTargets:
        configured_kbs = list(config.knowledge_bases or [])
        configured_knowledge_ids = list(config.knowledge_ids or [])

        if not configured_kbs and not configured_knowledge_ids:
            return SearchTargets()

        targets = SearchTargets()
        if configured_knowledge_ids:
            by_kb: dict[str, list[str]] = {}
            unresolved: list[str] = []
            for knowledge_id in configured_knowledge_ids:
                record = self.knowledge_records.get(knowledge_id)
                if record is None:
                    unresolved.append(knowledge_id)
                    continue
                by_kb.setdefault(record.knowledge_base_id, []).append(knowledge_id)
            for kb_id, ids in by_kb.items():
                if configured_kbs and kb_id not in configured_kbs:
                    continue
                kb = self.knowledge_bases.get(kb_id)
                targets.append(
                    SearchTarget(
                        knowledge_base_id=kb_id,
                        knowledge_ids=tuple(ids),
                        tenant_id=_target_tenant_id(kb, tenant_id),
                        target_type="knowledge",
                        knowledge_base_type=kb.kb_type if kb is not None else "",
                    )
                )
            if unresolved:
                for kb_id in configured_kbs:
                    kb = self.knowledge_bases.get(kb_id)
                    targets.append(
                        SearchTarget(
                            knowledge_base_id=kb_id,
                            knowledge_ids=tuple(unresolved),
                            tenant_id=_target_tenant_id(kb, tenant_id),
                            target_type="knowledge",
                            knowledge_base_type=kb.kb_type if kb is not None else "",
                        )
                    )
        else:
            for kb_id in configured_kbs:
                kb = self.knowledge_bases.get(kb_id)
                targets.append(
                    SearchTarget(
                        knowledge_base_id=kb_id,
                        tenant_id=_target_tenant_id(kb, tenant_id),
                        knowledge_base_type=kb.kb_type if kb is not None else "",
                    )
                )

        return SearchTargets(
            [target for target in targets if self._target_has_rag_capability(target)]
        )

    def _target_has_rag_capability(self, target: SearchTarget) -> bool:
        kb = self.knowledge_bases.get(target.knowledge_base_id)
        if kb is None:
            return True
        return kb.vector_enabled or kb.keyword_enabled

    def _ensure_knowledge_search_reranker_configured(self) -> None:
        if getattr(self.rerank_service, "reranker", None) is not None:
            return
        if not self._rerank_service_provided:
            self.rerank_service = default_rerank_service_from_env(required=True)
            if getattr(self.rerank_service, "reranker", None) is not None:
                return
        raise ValueError(
            "rerank model is not configured: knowledge_search requires an explicit reranker"
        )


class QueryService:
    def __init__(self, agent_service: AgentService) -> None:
        self.agent_service = agent_service

    def retrieve(self, request: QueryRequest) -> QueryResult:
        config = AgentConfig(
            knowledge_bases=list(request.knowledge_base_ids),
            knowledge_ids=list(request.knowledge_ids),
        )
        targets = self.agent_service.build_search_targets(config, tenant_id=request.tenant_id)
        knowledge_base_ids = targets.get_all_knowledge_base_ids()
        if not targets or not knowledge_base_ids:
            return QueryResult(
                request=request,
                pipeline_result=PipelineResult([], [], [], []),
            )

        pipeline = self.agent_service.create_search_pipeline(
            config,
            tenant_id=request.tenant_id,
        )
        result = pipeline.run(
            request.query,
            primary_knowledge_base_id=knowledge_base_ids[0],
            knowledge_base_ids=knowledge_base_ids,
            knowledge_ids=request.knowledge_ids,
            top_k=request.top_k,
            vector_threshold=request.vector_threshold,
            keyword_threshold=request.keyword_threshold,
            rerank_top_k=request.top_k,
            rerank_threshold=request.rerank_threshold,
            tenant_id=request.tenant_id if request.tenant_id else None,
        )
        return QueryResult(request=request, pipeline_result=result)


def config_with_scope(
    base: AgentConfig | None,
    *,
    knowledge_base_ids: Sequence[str],
    knowledge_ids: Sequence[str],
) -> AgentConfig:
    config = replace(base) if base is not None else AgentConfig()
    if knowledge_base_ids:
        config.knowledge_bases = list(knowledge_base_ids)
    if knowledge_ids:
        config.knowledge_ids = list(knowledge_ids)
    if not config.allowed_tools:
        config.allowed_tools = default_allowed_tools()
    return config


def default_rerank_service_from_env(*, required: bool = False) -> RerankService:
    enabled = os.getenv("RERANK_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "n", "off"}:
        if required:
            raise ValueError(
                "rerank model is not configured: RERANK_ENABLED disables rerank, "
                "but knowledge_search requires an explicit reranker"
            )
        return RerankService()
    if not (os.getenv("RERANK_API_KEY") or os.getenv("SILICONFLOW_API_KEY")):
        if required:
            raise ValueError(
                "rerank model is not configured: set RERANK_API_KEY or SILICONFLOW_API_KEY"
            )
        return RerankService()
    return RerankService(SiliconFlowReranker.from_env())


def _target_tenant_id(kb: KnowledgeBaseConfig | None, fallback: int) -> int:
    if kb is not None and kb.tenant_id:
        return kb.tenant_id
    return fallback


def _dedup(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
