from __future__ import annotations

from unittest import TestCase

from tiny_rag.indexing.models import (
    CHUNK_SOURCE_TYPE,
    IndexWithScore,
    KEYWORDS_RETRIEVER_TYPE,
    MATCH_TYPE_EMBEDDING,
    MATCH_TYPE_KEYWORDS,
    RetrieveResult,
    SQLITE_RETRIEVER_ENGINE_TYPE,
    VECTOR_RETRIEVER_TYPE,
)
from tiny_rag.retrieval import HybridSearchService, KnowledgeBaseRef, ResolvedEmbeddingModel


class FakeEmbedder:
    def __init__(self, name: str = "fake", model_id: str = "model"):
        self.name = name
        self.model_id = model_id
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0]

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def get_model_name(self) -> str:
        return self.name

    def get_dimensions(self) -> int:
        return 2

    def get_model_id(self) -> str:
        return self.model_id


class FakeRepository:
    def __init__(
        self,
        *,
        vector=None,
        keyword=None,
        support=None,
        engine_type: str = SQLITE_RETRIEVER_ENGINE_TYPE,
    ):
        self.vector = vector or []
        self.keyword = keyword or []
        self._support = support or [KEYWORDS_RETRIEVER_TYPE, VECTOR_RETRIEVER_TYPE]
        self.engine_type = engine_type
        self.params = []

    def support(self):
        return list(self._support)

    def retrieve(self, params):
        self.params.append(params)
        if params.retriever_type == VECTOR_RETRIEVER_TYPE:
            return [
                RetrieveResult(
                    results=list(self.vector),
                    retriever_engine_type=self.engine_type,
                    retriever_type=VECTOR_RETRIEVER_TYPE,
                )
            ]
        return [
            RetrieveResult(
                results=list(self.keyword),
                retriever_engine_type=self.engine_type,
                retriever_type=KEYWORDS_RETRIEVER_TYPE,
            )
        ]


def hit(chunk_id: str, score: float, match_type: int) -> IndexWithScore:
    return IndexWithScore(
        id=chunk_id,
        content=f"content {chunk_id}",
        source_id=f"source-{chunk_id}",
        source_type=CHUNK_SOURCE_TYPE,
        chunk_id=chunk_id,
        knowledge_id="knowledge-1",
        knowledge_base_id="kb-1",
        tag_id="",
        score=score,
        match_type=match_type,
    )


def kb(kb_id: str = "kb-1", *, model: str = "model", store: str = "", tenant: int = 7):
    return KnowledgeBaseRef(
        id=kb_id,
        tenant_id=tenant,
        embedding_model_id=model,
        embedding_model_key=model,
        vector_store_id=store,
    )


class HybridSearchServiceTests(TestCase):
    def test_vector_only_deduplicates_by_chunk_id_highest_score(self):
        repo = FakeRepository(
            vector=[
                hit("chunk-a", 0.2, MATCH_TYPE_EMBEDDING),
                hit("chunk-b", 0.8, MATCH_TYPE_EMBEDDING),
                hit("chunk-a", 0.9, MATCH_TYPE_EMBEDDING),
            ],
            support=[VECTOR_RETRIEVER_TYPE],
        )
        service = HybridSearchService(repo, embedder=FakeEmbedder(), knowledge_bases=[kb()])

        results = service.search("q", knowledge_base_ids=["kb-1"], disable_keywords_match=True)

        self.assertEqual([result.id for result in results], ["chunk-a", "chunk-b"])
        self.assertEqual([result.score for result in results], [0.9, 0.8])

    def test_keyword_only_deduplicates_by_chunk_id_highest_score(self):
        repo = FakeRepository(
            keyword=[
                hit("chunk-a", 1.0, MATCH_TYPE_KEYWORDS),
                hit("chunk-a", 3.0, MATCH_TYPE_KEYWORDS),
                hit("chunk-b", 2.0, MATCH_TYPE_KEYWORDS),
            ],
            support=[KEYWORDS_RETRIEVER_TYPE],
        )
        service = HybridSearchService(repo, embedder=FakeEmbedder(), knowledge_bases=[kb()])

        results = service.search("q", knowledge_base_ids=["kb-1"], disable_vector_match=True)

        self.assertEqual([result.id for result in results], ["chunk-a", "chunk-b"])
        self.assertEqual([result.score for result in results], [3.0, 2.0])

    def test_hybrid_uses_weighted_rrf(self):
        repo = FakeRepository(
            vector=[
                hit("chunk-a", 0.9, MATCH_TYPE_EMBEDDING),
                hit("chunk-b", 0.8, MATCH_TYPE_EMBEDDING),
            ],
            keyword=[
                hit("chunk-b", 10.0, MATCH_TYPE_KEYWORDS),
                hit("chunk-c", 9.0, MATCH_TYPE_KEYWORDS),
            ],
        )
        service = HybridSearchService(repo, embedder=FakeEmbedder(), knowledge_bases=[kb()])

        results = service.search("q", knowledge_base_ids=["kb-1"])

        self.assertEqual([result.id for result in results], ["chunk-b", "chunk-a", "chunk-c"])
        self.assertAlmostEqual(results[0].score, 0.7 / 62 + 0.3 / 61)
        self.assertAlmostEqual(results[1].score, 0.7 / 61)
        self.assertAlmostEqual(results[2].score, 0.3 / 62)

    def test_multi_kb_embedding_model_mismatch_raises(self):
        repo = FakeRepository(vector=[hit("chunk-a", 1.0, MATCH_TYPE_EMBEDDING)])
        service = HybridSearchService(
            repo,
            embedder=FakeEmbedder(),
            knowledge_bases=[kb("kb-1", model="m1"), kb("kb-2", model="m2")],
        )

        with self.assertRaisesRegex(ValueError, "different embedding models"):
            service.search("q", knowledge_base_ids=["kb-1", "kb-2"])

    def test_query_embedding_reused_across_store_groups(self):
        embedder = FakeEmbedder()
        repo_a = FakeRepository(vector=[hit("chunk-a", 0.9, MATCH_TYPE_EMBEDDING)])
        repo_b = FakeRepository(vector=[hit("chunk-b", 0.8, MATCH_TYPE_EMBEDDING)])
        repos = {"store-a": repo_a, "store-b": repo_b}

        def resolver(store_id, tenant_id, engine_type):
            return repos[store_id]

        service = HybridSearchService(
            repo_a,
            embedder=embedder,
            repository_resolver=resolver,
            knowledge_bases=[
                kb("kb-1", model="same", store="store-a"),
                kb("kb-2", model="same", store="store-b"),
            ],
        )

        results = service.search(
            "shared query",
            knowledge_base_ids=["kb-1", "kb-2"],
            disable_keywords_match=True,
        )

        self.assertEqual(embedder.calls, ["shared query"])
        self.assertEqual([result.id for result in results], ["chunk-a", "chunk-b"])
        self.assertEqual(repo_a.params[0].embedding, [1.0, 0.0])
        self.assertEqual(repo_b.params[0].embedding, [1.0, 0.0])

    def test_embedding_model_resolver_uses_source_tenant_model_identity_and_embedder(self):
        source_embedder = FakeEmbedder("source-model", "source-id")
        other_embedder = FakeEmbedder("other-model", "other-id")
        repo = FakeRepository(vector=[hit("chunk-a", 0.9, MATCH_TYPE_EMBEDDING)])
        resolver_calls = []

        def resolver(ref: KnowledgeBaseRef):
            resolver_calls.append((ref.embedding_model_id, ref.tenant_id))
            if ref.tenant_id == 42:
                return ResolvedEmbeddingModel("real-name|https://source", source_embedder)
            return ResolvedEmbeddingModel("real-name|https://source", other_embedder)

        service = HybridSearchService(
            repo,
            embedder=FakeEmbedder("global", "global"),
            embedding_model_resolver=resolver,
            knowledge_bases=[
                kb("kb-1", model="tenant-model-id", tenant=42),
                kb("kb-2", model="different-id-same-real-model", tenant=99),
            ],
        )

        service.search(
            "source tenant query",
            primary_knowledge_base_id="kb-1",
            knowledge_base_ids=["kb-1", "kb-2"],
            disable_keywords_match=True,
        )

        self.assertEqual(
            resolver_calls,
            [("tenant-model-id", 42), ("different-id-same-real-model", 99)],
        )
        self.assertEqual(source_embedder.calls, ["source tenant query"])
        self.assertEqual(other_embedder.calls, [])

    def test_fan_out_merges_results_from_store_groups(self):
        repo_a = FakeRepository(vector=[hit("chunk-a", 0.7, MATCH_TYPE_EMBEDDING)])
        repo_b = FakeRepository(vector=[hit("chunk-b", 0.9, MATCH_TYPE_EMBEDDING)])

        def resolver(store_id, tenant_id, engine_type):
            return {"store-a": repo_a, "store-b": repo_b}[store_id]

        service = HybridSearchService(
            repo_a,
            embedder=FakeEmbedder(),
            repository_resolver=resolver,
            knowledge_bases=[
                kb("kb-1", model="same", store="store-a"),
                kb("kb-2", model="same", store="store-b"),
            ],
        )

        results = service.search(
            "q",
            knowledge_base_ids=["kb-1", "kb-2"],
            disable_keywords_match=True,
        )

        self.assertEqual([result.id for result in results], ["chunk-b", "chunk-a"])
