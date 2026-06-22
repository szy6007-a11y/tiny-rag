from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from tiny_rag.agent import AgentConfig
from tiny_rag.agent.tools import ToolKnowledgeSearch
from tiny_rag.persistence import connect
from tiny_rag.retrieval import RankResult, RerankService, SiliconFlowReranker
from tiny_rag.service import (
    AgentService,
    KnowledgeBaseConfig,
    QueryRequest,
    QueryService,
    default_rerank_service_from_env,
)


class AgentServiceFactoryTests(TestCase):
    def test_default_rerank_service_is_disabled_without_api_key(self):
        with patch.dict(
            "os.environ",
            {
                "RERANK_ENABLED": "true",
                "RERANK_API_KEY": "",
                "SILICONFLOW_API_KEY": "",
            },
            clear=False,
        ):
            service = default_rerank_service_from_env()

        self.assertIsNone(service.reranker)

    def test_default_rerank_service_can_be_disabled_even_with_api_key(self):
        with patch.dict(
            "os.environ",
            {
                "RERANK_ENABLED": "false",
                "RERANK_API_KEY": "test-key",
                "SILICONFLOW_API_KEY": "",
            },
            clear=False,
        ):
            service = default_rerank_service_from_env()

        self.assertIsNone(service.reranker)

    def test_default_rerank_service_uses_siliconflow_when_api_key_exists(self):
        with patch.dict(
            "os.environ",
            {
                "RERANK_ENABLED": "true",
                "RERANK_API_KEY": "test-key",
                "SILICONFLOW_API_KEY": "",
                "RERANK_PROVIDER": "siliconflow",
                "RERANK_MODEL_NAME": "BAAI/bge-reranker-v2-m3",
            },
            clear=False,
        ):
            service = default_rerank_service_from_env()

        self.assertIsInstance(service.reranker, SiliconFlowReranker)

    def test_default_rerank_service_required_raises_without_api_key(self):
        with patch.dict(
            "os.environ",
            {
                "RERANK_ENABLED": "true",
                "RERANK_API_KEY": "",
                "SILICONFLOW_API_KEY": "",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "RERANK_API_KEY"):
                default_rerank_service_from_env(required=True)


class AgentServiceAssemblyTests(TestCase):
    def test_default_merge_service_uses_chunk_repository_and_request_tenant(self):
        conn = connect(":memory:")
        self.addCleanup(conn.close)
        kb = KnowledgeBaseConfig(id="kb-1", tenant_id=7)
        service = AgentService(
            conn,
            knowledge_bases={kb.id: kb},
            rerank_service=RerankService(PassthroughReranker()),
        )

        pipeline = service.create_search_pipeline(
            AgentConfig(knowledge_bases=[kb.id]),
            tenant_id=7,
        )

        self.assertIs(pipeline.merge_service.chunk_repository, service.chunk_repository)
        self.assertEqual(pipeline.merge_service.tenant_id, 7)

    def test_no_scope_does_not_expand_to_all_knowledge_bases(self):
        conn = connect(":memory:")
        self.addCleanup(conn.close)
        service = AgentService(
            conn,
            knowledge_bases={"kb-1": KnowledgeBaseConfig(id="kb-1", tenant_id=7)},
        )

        targets = service.build_search_targets(AgentConfig(), tenant_id=7)
        registry = service.create_tool_registry(AgentConfig(), tenant_id=7)

        self.assertEqual(targets, [])
        self.assertEqual(registry.list_tools(), [])

    def test_knowledge_search_requires_configured_reranker(self):
        conn = connect(":memory:")
        self.addCleanup(conn.close)
        with patch.dict(
            "os.environ",
            {
                "RERANK_ENABLED": "true",
                "RERANK_API_KEY": "",
                "SILICONFLOW_API_KEY": "",
            },
            clear=False,
        ):
            service = AgentService(
                conn,
                knowledge_bases={"kb-1": KnowledgeBaseConfig(id="kb-1", tenant_id=7)},
            )

            with self.assertRaisesRegex(ValueError, "RERANK_API_KEY"):
                service.create_tool_registry(
                    AgentConfig(
                        knowledge_bases=["kb-1"],
                        allowed_tools=[ToolKnowledgeSearch],
                    ),
                    tenant_id=7,
                )

    def test_knowledge_search_rerank_requirement_is_skipped_without_scope(self):
        conn = connect(":memory:")
        self.addCleanup(conn.close)
        with patch.dict(
            "os.environ",
            {
                "RERANK_ENABLED": "true",
                "RERANK_API_KEY": "",
                "SILICONFLOW_API_KEY": "",
            },
            clear=False,
        ):
            service = AgentService(
                conn,
                knowledge_bases={"kb-1": KnowledgeBaseConfig(id="kb-1", tenant_id=7)},
            )

            registry = service.create_tool_registry(
                AgentConfig(allowed_tools=[ToolKnowledgeSearch]),
                tenant_id=7,
            )

        self.assertEqual(registry.list_tools(), [])

    def test_query_service_empty_scope_returns_empty_pipeline_result(self):
        conn = connect(":memory:")
        self.addCleanup(conn.close)
        service = AgentService(
            conn,
            knowledge_bases={"kb-1": KnowledgeBaseConfig(id="kb-1", tenant_id=7)},
        )

        result = QueryService(service).retrieve(QueryRequest(query="anything", tenant_id=7))

        self.assertEqual(result.pipeline_result.search_results, [])
        self.assertEqual(result.pipeline_result.results, [])


class PassthroughReranker:
    def rerank(self, query, passages, **kwargs):
        return [
            RankResult(index=index, relevance_score=1.0)
            for index, _ in enumerate(passages)
        ]
