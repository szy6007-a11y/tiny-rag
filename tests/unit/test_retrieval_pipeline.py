from __future__ import annotations

from unittest import TestCase

from tiny_rag.retrieval import KnowledgeBaseRef, SearchPipeline, SearchResult


class SearchPipelineAssemblyTests(TestCase):
    def test_default_merge_service_uses_hybrid_chunk_repository(self):
        hybrid = FakeHybridSearch(tenant_id=7)

        pipeline = SearchPipeline(hybrid, FakeRerankService(), tenant_id=7)

        self.assertIs(pipeline.merge_service.chunk_repository, hybrid.chunk_repository)
        self.assertEqual(pipeline.merge_service.tenant_id, 7)

    def test_merge_receives_tenant_from_scoped_knowledge_base(self):
        hybrid = FakeHybridSearch(
            knowledge_bases={"kb-1": KnowledgeBaseRef(id="kb-1", tenant_id=7)}
        )
        merge_service = CapturingMergeService()
        pipeline = SearchPipeline(hybrid, FakeRerankService(), merge_service)

        pipeline.run(
            "query",
            primary_knowledge_base_id="kb-1",
            knowledge_base_ids=["kb-1"],
        )

        self.assertEqual(merge_service.tenant_ids, [7])


class FakeHybridSearch:
    def __init__(
        self,
        *,
        tenant_id: int = 0,
        knowledge_bases: dict[str, KnowledgeBaseRef] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.knowledge_bases = dict(knowledge_bases or {})
        self.chunk_repository = object()

    def search(self, query: str, **kwargs):
        return [
            SearchResult(
                id="chunk-1",
                content="content",
                knowledge_id="knowledge-1",
                knowledge_base_id="kb-1",
                score=0.9,
            )
        ]


class FakeRerankService:
    def rerank(self, query: str, results, **kwargs):
        return list(results)


class CapturingMergeService:
    def __init__(self) -> None:
        self.tenant_ids: list[int | None] = []

    def merge(self, **kwargs):
        self.tenant_ids.append(kwargs.get("tenant_id"))
        return list(
            kwargs.get("rerank_results") or kwargs.get("search_results") or []
        )
