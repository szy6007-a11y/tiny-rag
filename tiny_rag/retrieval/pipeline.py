from __future__ import annotations

from typing import Sequence

from tiny_rag.cancellation import CancellationToken

from .hybrid import HybridSearchService
from .merge import MergeService
from .models import PipelineResult, RetrievalConfig, SearchResult
from .rerank import RerankService


class SearchPipeline:
    def __init__(
        self,
        hybrid_search: HybridSearchService,
        rerank_service: RerankService | None = None,
        merge_service: MergeService | None = None,
        *,
        tenant_id: int = 0,
    ) -> None:
        self.hybrid_search = hybrid_search
        self.rerank_service = rerank_service or RerankService()
        self.tenant_id = tenant_id or int(getattr(hybrid_search, "tenant_id", 0) or 0)
        self.merge_service = merge_service or MergeService(
            getattr(hybrid_search, "chunk_repository", None),
            tenant_id=self.tenant_id,
        )

    def run(
        self,
        query: str,
        *,
        primary_knowledge_base_id: str | None = None,
        knowledge_base_ids: Sequence[str] | None = None,
        top_k: int = 10,
        threshold: float = 0.0,
        vector_threshold: float | None = None,
        keyword_threshold: float | None = None,
        knowledge_ids: Sequence[str] | None = None,
        tag_ids: Sequence[str] | None = None,
        exclude_knowledge_ids: Sequence[str] | None = None,
        exclude_chunk_ids: Sequence[str] | None = None,
        disable_vector_match: bool = False,
        disable_keywords_match: bool = False,
        query_embedding: Sequence[float] | None = None,
        retrieval_config: RetrievalConfig | None = None,
        rerank_top_k: int | None = None,
        rerank_threshold: float = 0.2,
        faq_priority_enabled: bool = False,
        faq_score_boost: float = 1.0,
        history_references: Sequence[SearchResult] | None = None,
        tenant_id: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> PipelineResult:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        search_results = self.hybrid_search.search(
            query,
            primary_knowledge_base_id=primary_knowledge_base_id,
            knowledge_base_ids=knowledge_base_ids,
            top_k=top_k,
            threshold=threshold,
            vector_threshold=vector_threshold,
            keyword_threshold=keyword_threshold,
            knowledge_ids=knowledge_ids,
            tag_ids=tag_ids,
            exclude_knowledge_ids=exclude_knowledge_ids,
            exclude_chunk_ids=exclude_chunk_ids,
            disable_vector_match=disable_vector_match,
            disable_keywords_match=disable_keywords_match,
            query_embedding=query_embedding,
            retrieval_config=retrieval_config,
            cancellation_token=cancellation_token,
        )
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        effective_rerank_top_k = rerank_top_k if rerank_top_k is not None else top_k
        rerank_results = self.rerank_service.rerank(
            query,
            search_results,
            top_k=effective_rerank_top_k,
            threshold=rerank_threshold,
            faq_priority_enabled=faq_priority_enabled,
            faq_score_boost=faq_score_boost,
            cancellation_token=cancellation_token,
        )
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        effective_tenant_id = self._resolve_tenant_id(
            tenant_id=tenant_id,
            primary_knowledge_base_id=primary_knowledge_base_id,
            knowledge_base_ids=knowledge_base_ids,
        )
        merge_results = self.merge_service.merge(
            search_results=search_results,
            rerank_results=rerank_results,
            history_references=history_references,
            query=query,
            tenant_id=effective_tenant_id,
            cancellation_token=cancellation_token,
        )
        results = filter_top_k(
            merge_results or rerank_results or search_results,
            effective_rerank_top_k,
        )
        return PipelineResult(
            search_results=search_results,
            rerank_results=rerank_results,
            merge_results=merge_results,
            results=results,
        )

    def _resolve_tenant_id(
        self,
        *,
        tenant_id: int | None,
        primary_knowledge_base_id: str | None,
        knowledge_base_ids: Sequence[str] | None,
    ) -> int:
        if tenant_id is not None:
            return tenant_id
        if self.tenant_id:
            return self.tenant_id

        kb_map = getattr(self.hybrid_search, "knowledge_bases", None)
        if not isinstance(kb_map, dict) or not kb_map:
            return 0

        scoped_ids = [item for item in list(knowledge_base_ids or []) if item]
        if not scoped_ids and primary_knowledge_base_id:
            scoped_ids = [primary_knowledge_base_id]

        tenant_ids = {
            int(getattr(kb_map[kb_id], "tenant_id", 0) or 0)
            for kb_id in scoped_ids
            if kb_id in kb_map
            and int(getattr(kb_map[kb_id], "tenant_id", 0) or 0) != 0
        }
        if len(tenant_ids) == 1:
            return next(iter(tenant_ids))
        return 0


def retrieve_pipeline(
    query: str,
    *,
    hybrid_search: HybridSearchService,
    rerank_service: RerankService | None = None,
    merge_service: MergeService | None = None,
    **kwargs,
) -> PipelineResult:
    return SearchPipeline(hybrid_search, rerank_service, merge_service).run(query, **kwargs)


def filter_top_k(results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
    out = list(results)
    if top_k > 0 and len(out) > top_k:
        return out[:top_k]
    return out
