from __future__ import annotations

import unittest

from tiny_rag.agent import ChatOptions, ChatResponse, FunctionCall, LLMToolCall, Message
from tiny_rag.agent.tools import (
    ToolGetDocumentInfo,
    ToolGrepChunks,
    ToolKnowledgeSearch,
    ToolListKnowledgeChunks,
)
from tiny_rag.chunking import SplitterConfig
from tiny_rag.persistence import connect
from tiny_rag.retrieval import RankResult, RerankService
from tiny_rag.service import (
    AgentQARequest,
    AgentService,
    IngestRequest,
    IngestService,
    KnowledgeBaseConfig,
    SessionService,
)


class ProductionServiceE2ETests(unittest.TestCase):
    def test_ingest_then_agent_query_runs_full_rag_tool_chain(self):
        conn = connect(":memory:")
        self.addCleanup(conn.close)
        embedder = DeterministicEmbedder()
        kb = KnowledgeBaseConfig(
            id="kb-e2e",
            tenant_id=7,
            name="Operations KB",
            embedding_model_id=embedder.get_model_id(),
            keyword_enabled=True,
            vector_enabled=True,
        )
        ingest_service = IngestService(conn, embedder=embedder, knowledge_bases={kb.id: kb})
        ingest_result = ingest_service.ingest_document(
            IngestRequest(
                tenant_id=7,
                knowledge_id="knowledge-e2e",
                knowledge_base_id=kb.id,
                title="Retrieval Operations Runbook",
                file_name="runbook.md",
                file_type="md",
                content=b"""# Retrieval Operations Runbook

## Ingest

The document parser should normalize tables before chunking begins.

| Metric | Value |
| --- | --- |
| latency_budget | 120ms |
| index_path | sqlite-vector |

## Search

Operators can retrieve the latency_budget term through keyword search after indexing.
Vector search should also return the stored chunk metadata.
""",
                chunk_config=SplitterConfig(
                    chunk_size=180,
                    chunk_overlap=30,
                    separators=["\n\n", "\n", ". "],
                    strategy="auto",
                ),
                knowledge_base=kb,
            )
        )

        self.assertEqual(
            ingest_result.index_stats.text_chunk_count,
            ingest_result.index_stats.vector_count,
        )
        self.assertGreater(ingest_result.persist_result.text_count, 0)

        agent_service = AgentService(
            conn,
            repository=ingest_service.repository,
            chunk_repository=ingest_service.chunk_repository,
            embedder=embedder,
            knowledge_bases=ingest_service.knowledge_bases,
            knowledge_records=ingest_service.knowledge_records,
            rerank_service=RerankService(PassthroughReranker()),
        )
        chat_model = ToolCallingChatModel()
        session_service = SessionService(agent_service, chat_model)

        result = session_service.agent_qa(
            AgentQARequest(
                query="What is the latency budget?",
                session_id="session-e2e",
                tenant_id=7,
                knowledge_base_ids=[kb.id],
            )
        )

        self.assertTrue(result.state.is_complete)
        self.assertIn("120ms", result.state.final_answer)
        self.assertEqual(len(result.state.round_steps), 2)
        self.assertEqual(result.state.round_steps[0].tool_calls[0].name, ToolKnowledgeSearch)
        tool_names = chat_model.calls[0][1]
        self.assertEqual(
            tool_names,
            [
                ToolGetDocumentInfo,
                ToolGrepChunks,
                ToolKnowledgeSearch,
                ToolListKnowledgeChunks,
            ],
        )
        self.assertIn("latency_budget", chat_model.observed_tool_content)


class ToolCallingChatModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], list[str]]] = []
        self.observed_tool_content = ""

    def chat(self, messages: list[Message], opts: ChatOptions) -> ChatResponse:
        tool_names = [tool["function"]["name"] for tool in opts.tools]
        self.calls.append((list(messages), tool_names))
        if len(self.calls) == 1:
            self.assert_runtime_context(messages[-1].content)
            return ChatResponse(
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call-search",
                        function=FunctionCall(
                            name=ToolKnowledgeSearch,
                            arguments=(
                                '{"queries":["latency budget configured '
                                'in the operations runbook"]}'
                            ),
                        ),
                    )
                ],
            )

        tool_message = messages[-1]
        assert tool_message.role == "tool"
        self.observed_tool_content = tool_message.content
        assert "latency_budget" in tool_message.content
        assert "120ms" in tool_message.content
        return ChatResponse(
            content="The latency_budget in the runbook is 120ms.",
            finish_reason="stop",
        )

    def assert_runtime_context(self, content: str) -> None:
        assert "<runtime_context" in content
        assert 'knowledge_base id="kb-e2e"' in content


class DeterministicEmbedder:
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions
        self.batch_calls: list[list[str]] = []
        self.embeddings: list[list[float]] = []

    def embed(self, text: str) -> list[float]:
        return vector_for_text(text, self.dimensions)

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        embeddings = [vector_for_text(text, self.dimensions) for text in texts]
        self.embeddings.extend(embeddings)
        return embeddings

    def get_model_name(self) -> str:
        return "deterministic-test"

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_model_id(self) -> str:
        return "deterministic-test"


class PassthroughReranker:
    def rerank(self, query, passages, **kwargs):
        return [
            RankResult(index=index, relevance_score=1.0)
            for index, _ in enumerate(passages)
        ]


def vector_for_text(text: str, dimensions: int) -> list[float]:
    lowered = text.lower()
    features = [
        1.0 if "latency" in lowered else 0.1,
        1.0 if "budget" in lowered or "latency_budget" in lowered else 0.1,
        1.0 if "120ms" in lowered else 0.1,
        float(len(text) % 101 + 1) / 101.0,
    ]
    if dimensions <= len(features):
        return features[:dimensions]
    return features + [0.1] * (dimensions - len(features))


if __name__ == "__main__":
    unittest.main()
