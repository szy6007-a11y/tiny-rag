from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from tiny_rag.cancellation import CancellationToken, call_with_cancellation
from tiny_rag.indexing.models import (
    KEYWORDS_RETRIEVER_TYPE,
    MATCH_TYPE_EMBEDDING,
    MATCH_TYPE_KEYWORDS,
    RetrieveParams,
    RetrieveResult,
    SQLITE_RETRIEVER_ENGINE_TYPE,
    VECTOR_RETRIEVER_TYPE,
)
from tiny_rag.persistence.chunk_models import ChunkRow

from .models import (
    CHUNK_TYPE_TEXT,
    EmbeddingModelResolver,
    KNOWLEDGE_BASE_TYPE_FAQ,
    KNOWLEDGE_TYPE_FAQ,
    EmbedderLike,
    KnowledgeBaseRef,
    RetrievalConfig,
    ResolvedEmbeddingModel,
    SearchResult,
)
from .utils import clamp01


RepositoryResolver = Callable[[str, int, str], object]


@dataclass
class _StoreGroup:
    store_id: str
    owner_tenant_id: int
    engine_type: str
    kb_ids: list[str]
    repository: object
    base_params: list[RetrieveParams]
    top_k: int


class EngineAwareNormalizer:
    def normalize(self, score: float, retriever_type: str, engine_type: str) -> float:
        if retriever_type != VECTOR_RETRIEVER_TYPE:
            return score
        if engine_type == "milvus":
            return clamp01((score + 1.0) / 2.0)
        return clamp01(score)


class HybridSearchService:
    def __init__(
        self,
        repository: object,
        *,
        embedder: EmbedderLike | None = None,
        chunk_repository: object | None = None,
        tenant_id: int = 0,
        knowledge_bases: Mapping[str, KnowledgeBaseRef] | Sequence[KnowledgeBaseRef] | None = None,
        repository_resolver: RepositoryResolver | None = None,
        embedding_model_resolver: EmbeddingModelResolver | Callable[[KnowledgeBaseRef], ResolvedEmbeddingModel | EmbedderLike | tuple[str, EmbedderLike | None] | str] | None = None,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.chunk_repository = chunk_repository
        self.tenant_id = tenant_id
        self.knowledge_bases = self._normalize_kb_map(knowledge_bases)
        self.repository_resolver = repository_resolver
        self.embedding_model_resolver = embedding_model_resolver

    def search(
        self,
        query: str,
        *,
        primary_knowledge_base_id: str | None = None,
        knowledge_base_ids: Sequence[str] | None = None,
        knowledge_bases: Mapping[str, KnowledgeBaseRef] | Sequence[KnowledgeBaseRef] | None = None,
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
        cancellation_token: CancellationToken | None = None,
    ) -> list[SearchResult]:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        final_top_k = top_k if top_k > 0 else 10
        kb_map = dict(self.knowledge_bases)
        kb_map.update(self._normalize_kb_map(knowledge_bases))

        search_kb_ids = list(knowledge_base_ids or [])
        if not search_kb_ids:
            if primary_knowledge_base_id:
                search_kb_ids = [primary_knowledge_base_id]
            elif kb_map:
                search_kb_ids = [next(iter(kb_map))]
            else:
                search_kb_ids = [""]

        scoped_kbs = [kb_map.get(kb_id) or self._default_kb(kb_id) for kb_id in search_kb_ids]
        primary_id = primary_knowledge_base_id or search_kb_ids[0]
        primary = next((kb for kb in scoped_kbs if kb.id == primary_id), None)
        if primary is None:
            raise ValueError("knowledge base not found")

        resolved_models = self._resolve_embedding_models(scoped_kbs)
        self._validate_same_embedding_model(scoped_kbs, resolved_models)

        match_count = max(final_top_k * 5, 50) * len(search_kb_ids)
        if match_count > 500:
            match_count = 500

        embedding_cache: dict[tuple[str, str], list[float]] = {}
        params_query_embedding = list(query_embedding or [])
        if (
            not params_query_embedding
            and primary.is_vector_enabled()
            and not disable_vector_match
            and resolved_models.get(primary.id, ResolvedEmbeddingModel("")).identity_key != ""
        ):
            params_query_embedding = self._resolve_query_embedding(
                query,
                resolved_models[primary.id],
                embedding_cache,
                cancellation_token=cancellation_token,
            )

        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        groups = self._resolve_store_groups(
            primary=primary,
            kbs=scoped_kbs,
            query=query,
            query_embedding=params_query_embedding,
            top_k=match_count,
            vector_threshold=threshold if vector_threshold is None else vector_threshold,
            keyword_threshold=threshold if keyword_threshold is None else keyword_threshold,
            knowledge_ids=list(knowledge_ids or []),
            tag_ids=list(tag_ids or []),
            exclude_knowledge_ids=list(exclude_knowledge_ids or []),
            exclude_chunk_ids=list(exclude_chunk_ids or []),
            disable_vector_match=disable_vector_match,
            disable_keywords_match=disable_keywords_match,
            embedding_cache=embedding_cache,
            resolved_models=resolved_models,
            cancellation_token=cancellation_token,
        )
        if not groups or all(len(group.base_params) == 0 for group in groups):
            return []

        retrieve_results = self._retrieve_from_stores(
            groups,
            EngineAwareNormalizer(),
            cancellation_token=cancellation_token,
        )
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        vector_results, keyword_results = classify_retrieval_results(retrieve_results)
        if not vector_results and not keyword_results:
            return []

        fused = fuse_or_deduplicate(vector_results, keyword_results, retrieval_config)
        if len(fused) > final_top_k:
            fused = fused[:final_top_k]
        return self._to_search_results(fused)

    def _default_kb(self, kb_id: str) -> KnowledgeBaseRef:
        model_id = ""
        if self.embedder is not None:
            try:
                model_id = self.embedder.get_model_id()
            except Exception:
                model_id = ""
        return KnowledgeBaseRef(id=kb_id, tenant_id=self.tenant_id, embedding_model_id=model_id)

    def _resolve_query_embedding(
        self,
        query: str,
        resolved_model: ResolvedEmbeddingModel,
        cache: dict[tuple[str, str], list[float]],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[float]:
        model_identity = resolved_model.identity_key
        key = (model_identity, query)
        if key in cache:
            return cache[key]
        embedder = resolved_model.embedder or self.embedder
        if embedder is None:
            raise ValueError("embedder is required when vector retrieval is enabled")
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        if hasattr(embedder, "embed"):
            embedding = list(
                call_with_cancellation(
                    embedder.embed,
                    query,
                    cancellation_token=cancellation_token,
                )
            )
        else:
            embedding = list(
                call_with_cancellation(
                    embedder.batch_embed,
                    [query],
                    cancellation_token=cancellation_token,
                )[0]
            )
        cache[key] = embedding
        return embedding

    def _resolve_store_groups(
        self,
        *,
        primary: KnowledgeBaseRef,
        kbs: Sequence[KnowledgeBaseRef],
        query: str,
        query_embedding: Sequence[float],
        top_k: int,
        vector_threshold: float,
        keyword_threshold: float,
        knowledge_ids: list[str],
        tag_ids: list[str],
        exclude_knowledge_ids: list[str],
        exclude_chunk_ids: list[str],
        disable_vector_match: bool,
        disable_keywords_match: bool,
        embedding_cache: dict[tuple[str, str], list[float]],
        resolved_models: Mapping[str, ResolvedEmbeddingModel],
        cancellation_token: CancellationToken | None = None,
    ) -> list[_StoreGroup]:
        buckets: dict[tuple[str, int, str], list[KnowledgeBaseRef]] = {}
        for kb in kbs:
            key = (kb.vector_store_id, kb.tenant_id, kb.retriever_engine_type)
            buckets.setdefault(key, []).append(kb)

        groups: list[_StoreGroup] = []
        for (store_id, tenant_id, engine_type), group_kbs in buckets.items():
            repository = self._resolve_repository(store_id, tenant_id, engine_type)
            base_params = self._build_retrieval_params(
                repository=repository,
                primary=primary,
                group_kbs=group_kbs,
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                vector_threshold=vector_threshold,
                keyword_threshold=keyword_threshold,
                knowledge_ids=knowledge_ids,
                tag_ids=tag_ids,
                exclude_knowledge_ids=exclude_knowledge_ids,
                exclude_chunk_ids=exclude_chunk_ids,
                disable_vector_match=disable_vector_match,
                disable_keywords_match=disable_keywords_match,
                embedding_cache=embedding_cache,
                resolved_models=resolved_models,
                cancellation_token=cancellation_token,
            )
            groups.append(
                _StoreGroup(
                    store_id=store_id,
                    owner_tenant_id=tenant_id,
                    engine_type=engine_type,
                    kb_ids=[kb.id for kb in group_kbs],
                    repository=repository,
                    base_params=base_params,
                    top_k=top_k,
                )
            )
        return groups

    def _resolve_repository(self, store_id: str, tenant_id: int, engine_type: str) -> object:
        if self.repository_resolver is not None:
            return self.repository_resolver(store_id, tenant_id, engine_type)
        return self.repository

    def _build_retrieval_params(
        self,
        *,
        repository: object,
        primary: KnowledgeBaseRef,
        group_kbs: Sequence[KnowledgeBaseRef],
        query: str,
        query_embedding: Sequence[float],
        top_k: int,
        vector_threshold: float,
        keyword_threshold: float,
        knowledge_ids: list[str],
        tag_ids: list[str],
        exclude_knowledge_ids: list[str],
        exclude_chunk_ids: list[str],
        disable_vector_match: bool,
        disable_keywords_match: bool,
        embedding_cache: dict[tuple[str, str], list[float]],
        resolved_models: Mapping[str, ResolvedEmbeddingModel],
        cancellation_token: CancellationToken | None = None,
    ) -> list[RetrieveParams]:
        supported = set(repository.support() if hasattr(repository, "support") else [])
        params: list[RetrieveParams] = []
        faq_vector_kb_ids: list[str] = []
        doc_vector_kb_ids: list[str] = []
        doc_keyword_kb_ids: list[str] = []

        for kb in group_kbs:
            resolved_model = resolved_models.get(kb.id, ResolvedEmbeddingModel(""))
            if kb.is_vector_enabled() and resolved_model.identity_key != "":
                if kb.kb_type == KNOWLEDGE_BASE_TYPE_FAQ:
                    faq_vector_kb_ids.append(kb.id)
                else:
                    doc_vector_kb_ids.append(kb.id)
            if kb.is_keyword_enabled() and kb.kb_type != KNOWLEDGE_BASE_TYPE_FAQ:
                doc_keyword_kb_ids.append(kb.id)

        if (
            VECTOR_RETRIEVER_TYPE in supported
            and not disable_vector_match
            and (faq_vector_kb_ids or doc_vector_kb_ids)
        ):
            embedding = list(query_embedding)
            if not embedding:
                embedding = self._resolve_query_embedding(
                    query,
                    resolved_models.get(primary.id, ResolvedEmbeddingModel("")),
                    embedding_cache,
                    cancellation_token=cancellation_token,
                )

            def append_vector(kb_ids: list[str], knowledge_type: str) -> None:
                params.append(
                    RetrieveParams(
                        query=query,
                        embedding=embedding,
                        knowledge_base_ids=kb_ids,
                        knowledge_ids=knowledge_ids,
                        tag_ids=tag_ids,
                        exclude_knowledge_ids=exclude_knowledge_ids,
                        exclude_chunk_ids=exclude_chunk_ids,
                        top_k=top_k,
                        threshold=vector_threshold,
                        retriever_type=VECTOR_RETRIEVER_TYPE,
                        knowledge_type=knowledge_type,
                    )
                )

            if doc_vector_kb_ids:
                append_vector(doc_vector_kb_ids, "")
            if faq_vector_kb_ids:
                append_vector(faq_vector_kb_ids, KNOWLEDGE_TYPE_FAQ)

        if (
            KEYWORDS_RETRIEVER_TYPE in supported
            and not disable_keywords_match
            and doc_keyword_kb_ids
        ):
            params.append(
                RetrieveParams(
                    query=query,
                    knowledge_base_ids=doc_keyword_kb_ids,
                    knowledge_ids=knowledge_ids,
                    tag_ids=tag_ids,
                    exclude_knowledge_ids=exclude_knowledge_ids,
                    exclude_chunk_ids=exclude_chunk_ids,
                    top_k=top_k,
                    threshold=keyword_threshold,
                    retriever_type=KEYWORDS_RETRIEVER_TYPE,
                )
            )
        return params

    def _retrieve_from_stores(
        self,
        groups: Sequence[_StoreGroup],
        normalizer: EngineAwareNormalizer,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[RetrieveResult]:
        if not groups:
            return []
        all_results: list[RetrieveResult] = []
        for group in groups:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            params = [self._param_with_top_k(param, group.top_k) for param in group.base_params]
            all_results.extend(
                _repository_retrieve(
                    group.repository,
                    params,
                    cancellation_token=cancellation_token,
                )
            )

        if has_mixed_engine_types(all_results):
            for rr in all_results:
                for hit in rr.results or []:
                    hit.score = normalizer.normalize(
                        hit.score, rr.retriever_type, rr.retriever_engine_type
                    )
        return all_results

    def _param_with_top_k(self, param: RetrieveParams, top_k: int) -> RetrieveParams:
        return RetrieveParams(
            query=param.query,
            embedding=list(param.embedding or []),
            knowledge_base_ids=list(param.knowledge_base_ids or []),
            knowledge_ids=list(param.knowledge_ids or []),
            tag_ids=list(param.tag_ids or []),
            exclude_knowledge_ids=list(param.exclude_knowledge_ids or []),
            exclude_chunk_ids=list(param.exclude_chunk_ids or []),
            top_k=top_k,
            threshold=param.threshold,
            knowledge_type=param.knowledge_type,
            additional_params=dict(param.additional_params or {}),
            retriever_type=param.retriever_type,
        )

    def _to_search_results(self, chunks: Sequence[object]) -> list[SearchResult]:
        chunk_map = self._load_chunks([chunk.chunk_id for chunk in chunks])
        results: list[SearchResult] = []
        for item in chunks:
            row = chunk_map.get(item.chunk_id)
            if row is None:
                content = item.content
                start_at = 0
                end_at = len(content)
                chunk_index = 0
                chunk_type = CHUNK_TYPE_TEXT
                parent_chunk_id = ""
                image_info = ""
                metadata = None
            else:
                content = row.content
                start_at = row.start_at
                end_at = row.end_at
                chunk_index = row.chunk_index
                chunk_type = row.chunk_type
                parent_chunk_id = row.parent_chunk_id
                image_info = row.image_info
                metadata = row.metadata
            results.append(
                SearchResult(
                    id=item.chunk_id,
                    content=content,
                    knowledge_id=item.knowledge_id,
                    knowledge_base_id=item.knowledge_base_id,
                    chunk_index=chunk_index,
                    seq=chunk_index,
                    start_at=start_at,
                    end_at=end_at,
                    score=item.score,
                    match_type=item.match_type,
                    chunk_type=chunk_type,
                    parent_chunk_id=parent_chunk_id,
                    image_info=image_info,
                    chunk_metadata=metadata,
                    matched_content=item.content,
                    tag_id=item.tag_id,
                    source_id=item.source_id,
                    source_type=item.source_type,
                )
            )
        return results

    def _load_chunks(self, chunk_ids: Sequence[str]) -> dict[str, ChunkRow]:
        if self.chunk_repository is None or not chunk_ids:
            return {}
        if not hasattr(self.chunk_repository, "list_chunks_by_id"):
            return {}
        chunks = self.chunk_repository.list_chunks_by_id(
            tenant_id=self.tenant_id,
            chunk_ids=list(dict.fromkeys(chunk_ids)),
        )
        return {chunk.id: chunk for chunk in chunks}

    def _validate_same_embedding_model(
        self,
        kbs: Sequence[KnowledgeBaseRef],
        resolved_models: Mapping[str, ResolvedEmbeddingModel],
    ) -> None:
        if len(kbs) <= 1:
            return
        seen = ""
        for kb in kbs:
            key = resolved_models.get(kb.id, ResolvedEmbeddingModel("")).identity_key
            if key == "":
                continue
            if seen == "":
                seen = key
                continue
            if key != seen:
                raise ValueError(
                    "selected knowledge bases use different embedding models; "
                    "multi-KB search requires every knowledge base to share a single embedding model"
                )

    def _resolve_embedding_models(
        self, kbs: Sequence[KnowledgeBaseRef]
    ) -> dict[str, ResolvedEmbeddingModel]:
        resolved_by_ref: dict[tuple[str, int], ResolvedEmbeddingModel] = {}
        result: dict[str, ResolvedEmbeddingModel] = {}
        for kb in kbs:
            ref = (kb.embedding_model_id, kb.tenant_id)
            if ref not in resolved_by_ref:
                resolved_by_ref[ref] = self._resolve_embedding_model(kb)
            result[kb.id] = resolved_by_ref[ref]
        return result

    def _resolve_embedding_model(self, kb: KnowledgeBaseRef) -> ResolvedEmbeddingModel:
        resolver = self.embedding_model_resolver
        if resolver is not None:
            if hasattr(resolver, "resolve_embedding_model"):
                raw = resolver.resolve_embedding_model(kb)
            else:
                raw = resolver(kb)
            return self._coerce_resolved_embedding_model(raw, kb)
        return self._fallback_resolved_embedding_model(kb)

    def _coerce_resolved_embedding_model(
        self,
        raw: ResolvedEmbeddingModel | EmbedderLike | tuple[str, EmbedderLike | None] | str,
        kb: KnowledgeBaseRef,
    ) -> ResolvedEmbeddingModel:
        if isinstance(raw, ResolvedEmbeddingModel):
            return raw
        if isinstance(raw, str):
            return ResolvedEmbeddingModel(raw, self.embedder)
        if isinstance(raw, tuple):
            identity, embedder = raw
            return ResolvedEmbeddingModel(str(identity), embedder)
        identity = self._embedder_identity(raw, kb)
        return ResolvedEmbeddingModel(identity, raw)

    def _fallback_resolved_embedding_model(self, kb: KnowledgeBaseRef) -> ResolvedEmbeddingModel:
        identity = kb.model_identity()
        if identity == "" and self.embedder is not None:
            identity = self._embedder_identity(self.embedder, kb)
        return ResolvedEmbeddingModel(identity, self.embedder)

    def _embedder_identity(self, embedder: EmbedderLike | object, kb: KnowledgeBaseRef) -> str:
        if kb.embedding_model_key:
            return kb.embedding_model_key
        parts: list[str] = []
        for method_name in ("get_model_name", "get_model_id"):
            method = getattr(embedder, method_name, None)
            if method is None:
                continue
            try:
                value = method()
            except Exception:
                value = ""
            if value:
                parts.append(str(value))
        if parts:
            return "|".join(parts)
        return kb.embedding_model_id

    def _normalize_kb_map(
        self,
        knowledge_bases: Mapping[str, KnowledgeBaseRef] | Sequence[KnowledgeBaseRef] | None,
    ) -> dict[str, KnowledgeBaseRef]:
        if knowledge_bases is None:
            return {}
        if isinstance(knowledge_bases, Mapping):
            return dict(knowledge_bases)
        return {kb.id: kb for kb in knowledge_bases}


def _repository_retrieve(
    repository: object,
    params: Sequence[RetrieveParams],
    *,
    cancellation_token: CancellationToken | None = None,
) -> list[RetrieveResult]:
    if not params:
        return []
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    if hasattr(repository, "retrieve_many"):
        return list(
            call_with_cancellation(
                repository.retrieve_many,
                list(params),
                cancellation_token=cancellation_token,
            )
        )
    results: list[RetrieveResult] = []
    for param in params:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        result = call_with_cancellation(
            repository.retrieve,
            param,
            cancellation_token=cancellation_token,
        )
        results.extend(result)
    return results


def classify_retrieval_results(
    retrieve_results: Sequence[RetrieveResult],
) -> tuple[list[object], list[object]]:
    vector_results: list[object] = []
    keyword_results: list[object] = []
    for retrieve_result in retrieve_results:
        if retrieve_result.error is not None:
            raise retrieve_result.error
        items = list(retrieve_result.results or [])
        if retrieve_result.retriever_type == VECTOR_RETRIEVER_TYPE:
            vector_results.extend(items)
        else:
            keyword_results.extend(items)
    return vector_results, keyword_results


def fuse_or_deduplicate(
    vector_results: Sequence[object],
    keyword_results: Sequence[object],
    retrieval_config: RetrievalConfig | None = None,
) -> list[object]:
    if not keyword_results:
        return deduplicate_by_score(vector_results)
    if not vector_results:
        return deduplicate_by_score(keyword_results)
    return fuse_with_rrf(vector_results, keyword_results, retrieval_config)


def deduplicate_by_score(results: Sequence[object]) -> list[object]:
    chunk_info: dict[str, object] = {}
    for result in results:
        existing = chunk_info.get(result.chunk_id)
        if existing is None or result.score > existing.score:
            chunk_info[result.chunk_id] = result
    return sorted(chunk_info.values(), key=lambda item: item.score, reverse=True)


def fuse_with_rrf(
    vector_results: Sequence[object],
    keyword_results: Sequence[object],
    retrieval_config: RetrievalConfig | None = None,
) -> list[object]:
    cfg = retrieval_config or RetrievalConfig()
    rrf_k = cfg.effective_rrf_k()
    vector_weight, keyword_weight = cfg.effective_rrf_weights()

    vector_ranks: dict[str, int] = {}
    for index, result in enumerate(vector_results):
        vector_ranks.setdefault(result.chunk_id, index + 1)

    keyword_ranks: dict[str, int] = {}
    for index, result in enumerate(keyword_results):
        keyword_ranks.setdefault(result.chunk_id, index + 1)

    chunk_info: dict[str, object] = {}
    for result in vector_results:
        existing = chunk_info.get(result.chunk_id)
        if existing is None or result.score > existing.score:
            chunk_info[result.chunk_id] = result
    for result in keyword_results:
        chunk_info.setdefault(result.chunk_id, result)

    fused: list[object] = []
    for chunk_id, result in chunk_info.items():
        score = 0.0
        if chunk_id in vector_ranks:
            score += vector_weight / float(rrf_k + vector_ranks[chunk_id])
        if chunk_id in keyword_ranks:
            score += keyword_weight / float(rrf_k + keyword_ranks[chunk_id])
        result.score = score
        result.match_type = MATCH_TYPE_EMBEDDING if chunk_id in vector_ranks else MATCH_TYPE_KEYWORDS
        fused.append(result)
    return sorted(fused, key=lambda item: item.score, reverse=True)


def has_mixed_engine_types(results: Sequence[RetrieveResult]) -> bool:
    if len(results) < 2:
        return False
    first = results[0].retriever_engine_type
    return any(result.retriever_engine_type != first for result in results[1:])
