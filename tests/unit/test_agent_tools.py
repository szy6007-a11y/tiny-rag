from __future__ import annotations

import time
from types import SimpleNamespace
from unittest import TestCase

from tiny_rag.agent import (
    AgentStep,
    ChatResponse,
    FunctionCall,
    LLMToolCall,
    execute_tool_calls,
    run_tool_call,
)
from tiny_rag.agent.tools import (
    BaseTool,
    GetDocumentInfoTool,
    GrepChunksTool,
    ToolGetDocumentInfo,
    ToolGrepChunks,
    KnowledgeSearchTool,
    ListKnowledgeChunksTool,
    ToolListKnowledgeChunks,
    SearchTarget,
    ToolKnowledgeSearch,
    ToolRegistry,
    ToolResult,
    available_tool_definitions,
    default_allowed_tools,
    toolErrorHint,
)
from tiny_rag.agent.tools.tool import ToolExecutionError
from tiny_rag.persistence import ChunkRepository, connect, persist_text_chunks
from tiny_rag.retrieval.models import KnowledgeBaseRef, SearchResult


class DummyTool(BaseTool):
    def __init__(
        self,
        name: str,
        schema=None,
        result: ToolResult | None = None,
        *,
        sleep_seconds: float = 0,
    ) -> None:
        super().__init__(
            name,
            f"{name} description",
            schema
            or {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
            },
        )
        self.result = result or ToolResult(success=True, output=f"{name} output")
        self.sleep_seconds = sleep_seconds
        self.calls = []

    def execute(self, args):
        self.calls.append(args)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return self.result


class FakePipeline:
    def __init__(
        self,
        results_by_query: dict[str, list[SearchResult]],
        *,
        hybrid_search=None,
        reject_mixed_models: bool = False,
        fail_for_kbs: set[tuple[str, ...]] | None = None,
    ) -> None:
        self.results_by_query = results_by_query
        self.hybrid_search = hybrid_search
        self.reject_mixed_models = reject_mixed_models
        self.fail_for_kbs = fail_for_kbs or set()
        self.calls = []

    def run(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        knowledge_base_ids = tuple(kwargs.get("knowledge_base_ids") or [])
        if knowledge_base_ids in self.fail_for_kbs:
            raise RuntimeError(f"pipeline failed for {','.join(knowledge_base_ids)}")
        if self.reject_mixed_models:
            kb_map = getattr(self.hybrid_search, "knowledge_bases", {}) or {}
            models = {
                kb_map[kb_id].model_identity()
                for kb_id in knowledge_base_ids
                if kb_id in kb_map and kb_map[kb_id].model_identity()
            }
            if len(models) > 1:
                raise ValueError("selected knowledge bases use different embedding models")
        return SimpleNamespace(results=[item.clone() for item in self.results_by_query.get(query, [])])


def result(
    chunk_id: str,
    content: str,
    *,
    score: float = 0.5,
    knowledge_id: str = "knowledge-1",
    knowledge_base_id: str = "kb-1",
    chunk_index: int = 0,
    title: str = "Manual",
):
    return SearchResult(
        id=chunk_id,
        content=content,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_title=title,
        chunk_index=chunk_index,
        score=score,
    )


class ToolRegistryTests(TestCase):
    def test_stable_sorting_and_function_definitions(self):
        registry = ToolRegistry()
        first = DummyTool("b_tool", result=ToolResult(success=True, output="first"))
        duplicate = DummyTool("b_tool", result=ToolResult(success=True, output="duplicate"))
        registry.register_tool(first)
        registry.register_tool(DummyTool("a_tool"))
        registry.register_tool(duplicate)

        self.assertEqual(registry.list_tools(), ["a_tool", "b_tool"])
        definitions = registry.get_function_definitions()
        self.assertEqual([definition.name for definition in definitions], ["a_tool", "b_tool"])
        self.assertEqual(definitions[0].to_dict()["parameters"]["required"], ["query"])

        executed = registry.execute_tool("b_tool", {"query": "hello"})
        self.assertEqual(executed.output, "first")
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(duplicate.calls), 0)

    def test_execute_unknown_tool_returns_weknora_hint(self):
        result_value = ToolRegistry().execute_tool("missing_tool", {})

        self.assertFalse(result_value.success)
        self.assertEqual(result_value.output, "")
        self.assertIn("tool not found: missing_tool", result_value.error)
        self.assertTrue(result_value.error.endswith(toolErrorHint))

    def test_parameter_validation_failure_does_not_execute_tool(self):
        tool = DummyTool("needs_query")
        registry = ToolRegistry()
        registry.register_tool(tool)

        result_value = registry.execute_tool("needs_query", {"query": ""})

        self.assertFalse(result_value.success)
        self.assertIn("Parameter validation failed", result_value.error)
        self.assertIn("at least 1 characters", result_value.error)
        self.assertTrue(result_value.error.endswith(toolErrorHint))
        self.assertEqual(tool.calls, [])

    def test_casts_parameters_before_execution(self):
        schema = {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}, "patterns": {"type": "array"}},
            "required": ["enabled"],
        }
        tool = DummyTool("cast_tool", schema=schema)
        registry = ToolRegistry()
        registry.register_tool(tool)

        result_value = registry.execute_tool(
            "cast_tool", {"enabled": "true", "patterns": "OpenClaw"}
        )

        self.assertTrue(result_value.success)
        self.assertEqual(tool.calls[0]["enabled"], True)
        self.assertEqual(tool.calls[0]["patterns"], ["OpenClaw"])

    def test_validates_array_min_max_and_item_type(self):
        schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 2,
                }
            },
            "required": ["queries"],
        }
        tool = DummyTool("array_tool", schema=schema)
        registry = ToolRegistry()
        registry.register_tool(tool)

        too_few = registry.execute_tool("array_tool", {"queries": []})
        too_many = registry.execute_tool("array_tool", {"queries": ["a", "b", "c"]})
        bad_item = registry.execute_tool("array_tool", {"queries": [123]})

        self.assertIn("parameter 'queries' must have at least 1 items", too_few.error)
        self.assertIn("parameter 'queries' must have at most 2 items", too_many.error)
        self.assertIn("parameter 'queries[0]' should be type 'string'", bad_item.error)
        self.assertEqual(tool.calls, [])

    def test_output_truncation_preserves_registry_data(self):
        tool = DummyTool("large", result=ToolResult(success=True, output="H" * 50 + "T" * 50))
        registry = ToolRegistry()
        registry.set_max_tool_output_size(20)
        registry.register_tool(tool)

        result_value = registry.execute_tool("large", {"query": "ok"})

        self.assertEqual(result_value.output, "H" * 20)

    def test_default_exposed_tools_only_include_implemented_tools(self):
        self.assertEqual(
            default_allowed_tools(),
            [
                ToolKnowledgeSearch,
                ToolGrepChunks,
                ToolListKnowledgeChunks,
                ToolGetDocumentInfo,
            ],
        )
        self.assertEqual(
            [item["name"] for item in available_tool_definitions()],
            [
                ToolGrepChunks,
                ToolKnowledgeSearch,
                ToolListKnowledgeChunks,
                ToolGetDocumentInfo,
            ],
        )

    def test_malformed_json_arguments_are_repaired_before_execution(self):
        tool = DummyTool("repair_tool")
        registry = ToolRegistry()
        registry.register_tool(tool)

        tool_call = run_tool_call(
            registry,
            LLMToolCall(
                id="repair-1",
                function=FunctionCall(
                    name="repair_tool",
                    arguments='"query":"C\\+\\+",',
                ),
            ),
            0,
        )

        self.assertTrue(tool_call.result.success)
        self.assertEqual(tool.calls[0]["query"], "C\\+\\+")

    def test_tool_call_timeout_returns_failure_without_waiting_for_tool(self):
        tool = DummyTool(
            "slow_tool",
            schema={"type": "object", "properties": {}, "required": []},
            sleep_seconds=0.2,
        )
        registry = ToolRegistry()
        registry.register_tool(tool)
        response = ChatResponse(
            finish_reason="tool_calls",
            tool_calls=[
                LLMToolCall(
                    id="slow-1",
                    function=FunctionCall(name="slow_tool", arguments="{}"),
                )
            ],
        )
        step = AgentStep(iteration=0)

        started = time.monotonic()
        execute_tool_calls(registry, response, step, tool_call_timeout=0.01)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.15)
        self.assertEqual(len(step.tool_calls), 1)
        self.assertFalse(step.tool_calls[0].result.success)
        self.assertIn("timed out", step.tool_calls[0].result.error)
        self.assertTrue(step.tool_calls[0].result.error.endswith(toolErrorHint))


class KnowledgeSearchToolTests(TestCase):
    def test_function_definition_schema_is_weknora_knowledge_search(self):
        pipeline = FakePipeline({})
        tool = KnowledgeSearchTool(pipeline, allowed_knowledge_base_ids=["kb-1"])
        registry = ToolRegistry()
        registry.register_tool(tool)

        definitions = registry.get_function_definitions()

        self.assertEqual(definitions[0].name, ToolKnowledgeSearch)
        self.assertEqual(definitions[0].parameters["required"], ["queries"])
        self.assertEqual(definitions[0].parameters["properties"]["queries"]["maxItems"], 5)

    def test_registry_rejects_too_many_queries_and_non_string_items(self):
        pipeline = FakePipeline({})
        tool = KnowledgeSearchTool(pipeline, allowed_knowledge_base_ids=["kb-1"])
        registry = ToolRegistry()
        registry.register_tool(tool)

        too_many = registry.execute_tool(
            ToolKnowledgeSearch,
            {"queries": ["q1", "q2", "q3", "q4", "q5", "q6"]},
        )
        bad_item = registry.execute_tool(ToolKnowledgeSearch, {"queries": [123]})

        self.assertFalse(too_many.success)
        self.assertIn("parameter 'queries' must have at most 5 items", too_many.error)
        self.assertFalse(bad_item.success)
        self.assertIn("parameter 'queries[0]' should be type 'string'", bad_item.error)
        self.assertEqual(pipeline.calls, [])


    def test_direct_execute_strips_queries_and_knowledge_base_ids(self):
        pipeline = FakePipeline({"query": [result("chunk-1", "content", knowledge_base_id="kb-1")]})
        tool = KnowledgeSearchTool(pipeline, allowed_knowledge_base_ids=["kb-1"])

        result_value = tool.execute(
            {
                "queries": ["  query  ", "   "],
                "knowledge_base_ids": [" kb-1 ", "   "],
            }
        )

        self.assertTrue(result_value.success)
        self.assertEqual([call[0] for call in pipeline.calls], ["query"])
        self.assertEqual(pipeline.calls[0][1]["knowledge_base_ids"], ["kb-1"])
        self.assertEqual(result_value.data["queries"], ["query"])

    def test_direct_execute_rejects_too_many_queries(self):
        pipeline = FakePipeline({})
        tool = KnowledgeSearchTool(pipeline, allowed_knowledge_base_ids=["kb-1"])

        with self.assertRaisesRegex(
            ToolExecutionError, "queries parameter must contain at most 5 items"
        ):
            tool.execute({"queries": ["q1", "q2", "q3", "q4", "q5", "q6"]})

        self.assertEqual(pipeline.calls, [])

    def test_multi_query_scope_filtering_dedup_xml_and_data_results(self):
        pipeline = FakePipeline(
            {
                "What is RAG?": [
                    result(
                        "chunk-1",
                        "RAG combines retrieval with generation.",
                        score=0.7,
                        knowledge_id="doc-1",
                        knowledge_base_id="kb-1",
                    ),
                    result(
                        "chunk-out",
                        "This result is outside the runtime scope.",
                        score=0.99,
                        knowledge_id="doc-out",
                        knowledge_base_id="kb-3",
                    ),
                ],
                "How does retrieval help?": [
                    result(
                        "chunk-1-dup",
                        "  RAG   combines retrieval with generation.  ",
                        score=0.9,
                        knowledge_id="doc-1",
                        knowledge_base_id="kb-1",
                    ),
                    result(
                        "chunk-2",
                        "Retrieval provides grounded context for answers.",
                        score=0.8,
                        knowledge_id="doc-2",
                        knowledge_base_id="kb-2",
                        chunk_index=3,
                    ),
                ],
            }
        )
        tool = KnowledgeSearchTool(
            pipeline,
            search_targets=[
                SearchTarget("kb-1", knowledge_base_type="document"),
                SearchTarget("kb-2", knowledge_base_type="document"),
            ],
        )

        result_value = tool.execute(
            {
                "queries": ["What is RAG?", "How does retrieval help?"],
                "knowledge_base_ids": ["kb-1", "kb-2", "kb-3"],
            }
        )

        self.assertTrue(result_value.success)
        self.assertEqual([call[0] for call in pipeline.calls], ["What is RAG?", "How does retrieval help?"])
        for _, kwargs in pipeline.calls:
            self.assertEqual(kwargs["knowledge_base_ids"], ["kb-1", "kb-2"])

        self.assertIn('<search_results count="2">', result_value.output)
        self.assertIn("<query>What is RAG?</query>", result_value.output)
        self.assertIn('chunk_id="chunk-2"', result_value.output)
        self.assertIn("<match_snippet>", result_value.output)
        self.assertIn("<content>Retrieval provides grounded context for answers.</content>", result_value.output)
        self.assertNotIn("chunk-out", result_value.output)
        self.assertNotIn("chunk-1-dup", result_value.output)

        data = result_value.data or {}
        self.assertEqual(data["knowledge_base_ids"], ["kb-1", "kb-2"])
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["display_type"], "search_results")
        self.assertEqual(data["queries"], ["What is RAG?", "How does retrieval help?"])
        self.assertEqual([item["chunk_id"] for item in data["results"]], ["chunk-2", "chunk-1"])
        self.assertEqual(data["results"][0]["source_query"], "How does retrieval help?")
        self.assertEqual(data["results"][0]["query_type"], "hybrid")
        self.assertEqual(data["results"][0]["knowledge_base_type"], "document")

    def test_full_kb_targets_are_grouped_by_embedding_model(self):
        hybrid_search = SimpleNamespace(
            knowledge_bases={
                "kb-1": KnowledgeBaseRef(
                    id="kb-1", embedding_model_id="model-a", embedding_model_key="model-a"
                ),
                "kb-2": KnowledgeBaseRef(
                    id="kb-2", embedding_model_id="model-b", embedding_model_key="model-b"
                ),
            }
        )
        pipeline = FakePipeline(
            {
                "query": [
                    result("chunk-1", "alpha", score=0.9, knowledge_base_id="kb-1"),
                    result(
                        "chunk-2",
                        "beta",
                        score=0.8,
                        knowledge_id="knowledge-2",
                        knowledge_base_id="kb-2",
                    ),
                ]
            },
            hybrid_search=hybrid_search,
            reject_mixed_models=True,
        )
        tool = KnowledgeSearchTool(
            pipeline,
            search_targets=[SearchTarget("kb-1"), SearchTarget("kb-2")],
        )

        result_value = tool.execute({"queries": ["query"]})

        self.assertTrue(result_value.success)
        self.assertEqual(
            [call[1]["knowledge_base_ids"] for call in pipeline.calls],
            [["kb-1"], ["kb-2"]],
        )
        self.assertEqual([item["chunk_id"] for item in result_value.data["results"]], ["chunk-1", "chunk-2"])

    def test_multi_query_results_are_globally_limited_to_top_k(self):
        pipeline = FakePipeline(
            {
                f"query-{i}": [
                    result(
                        f"chunk-{i}",
                        f"content {i}",
                        score=1.0 - i * 0.1,
                        knowledge_id=f"knowledge-{i}",
                    )
                ]
                for i in range(5)
            }
        )
        tool = KnowledgeSearchTool(pipeline, allowed_knowledge_base_ids=["kb-1"], top_k=3)

        result_value = tool.execute({"queries": [f"query-{i}" for i in range(5)]})

        self.assertTrue(result_value.success)
        self.assertEqual(result_value.data["count"], 3)
        self.assertIn('<search_results count="3">', result_value.output)
        self.assertEqual(
            [item["chunk_id"] for item in result_value.data["results"]],
            ["chunk-0", "chunk-1", "chunk-2"],
        )

    def test_scope_filter_cannot_expand_permission(self):
        pipeline = FakePipeline({"query": [result("chunk-2", "secret", knowledge_base_id="kb-2")]})
        tool = KnowledgeSearchTool(pipeline, allowed_knowledge_base_ids=["kb-1"])
        registry = ToolRegistry()
        registry.register_tool(tool)

        result_value = registry.execute_tool(
            ToolKnowledgeSearch, {"queries": ["query"], "knowledge_base_ids": ["kb-2"]}
        )

        self.assertFalse(result_value.success)
        self.assertIn("no knowledge bases specified", result_value.error)
        self.assertEqual(pipeline.calls, [])

    def test_no_results_keeps_weknora_data_shape(self):
        pipeline = FakePipeline({"query": []})
        tool = KnowledgeSearchTool(pipeline, allowed_knowledge_base_ids=["kb-1"])

        result_value = tool.execute({"queries": ["query"]})

        self.assertTrue(result_value.success)
        self.assertIn("No relevant content found in 1 knowledge base(s).", result_value.output)
        self.assertEqual(result_value.data["results"], [])
        self.assertEqual(result_value.data["count"], 0)

    def test_knowledge_search_branch_failure_keeps_other_branch_results(self):
        hybrid_search = SimpleNamespace(
            knowledge_bases={
                "kb-1": KnowledgeBaseRef(
                    id="kb-1", embedding_model_id="model-a", embedding_model_key="model-a"
                ),
                "kb-2": KnowledgeBaseRef(
                    id="kb-2", embedding_model_id="model-b", embedding_model_key="model-b"
                ),
            }
        )
        pipeline = FakePipeline(
            {
                "query": [
                    result(
                        "chunk-2",
                        "surviving branch",
                        score=0.9,
                        knowledge_id="knowledge-2",
                        knowledge_base_id="kb-2",
                    )
                ]
            },
            hybrid_search=hybrid_search,
            fail_for_kbs={("kb-1",)},
        )
        tool = KnowledgeSearchTool(
            pipeline,
            search_targets=[SearchTarget("kb-1"), SearchTarget("kb-2")],
        )

        result_value = tool.execute({"queries": ["query"]})

        self.assertTrue(result_value.success)
        self.assertEqual(result_value.data["count"], 1)
        self.assertEqual(result_value.data["results"][0]["chunk_id"], "chunk-2")
        self.assertEqual(len(result_value.data["branch_failures"]), 1)
        self.assertIn("pipeline failed for kb-1", result_value.output)

    def test_knowledge_search_all_branches_fail_is_not_reported_as_no_results(self):
        hybrid_search = SimpleNamespace(
            knowledge_bases={
                "kb-1": KnowledgeBaseRef(
                    id="kb-1", embedding_model_id="model-a", embedding_model_key="model-a"
                ),
                "kb-2": KnowledgeBaseRef(
                    id="kb-2", embedding_model_id="model-b", embedding_model_key="model-b"
                ),
            }
        )
        pipeline = FakePipeline(
            {"query": []},
            hybrid_search=hybrid_search,
            fail_for_kbs={("kb-1",), ("kb-2",)},
        )
        tool = KnowledgeSearchTool(
            pipeline,
            search_targets=[SearchTarget("kb-1"), SearchTarget("kb-2")],
        )

        result_value = tool.execute({"queries": ["query"]})

        self.assertFalse(result_value.success)
        self.assertIn("All retrieval branches failed", result_value.error)
        self.assertNotIn("No relevant content found", result_value.output)
        self.assertEqual(result_value.data["count"], 0)
        self.assertEqual(len(result_value.data["branch_failures"]), 2)


class RetrievalInspectionToolTests(TestCase):
    def test_grep_list_and_document_info_share_search_scope(self):
        conn = connect(":memory:")
        persist_text_chunks(
            conn,
            tenant_id=9,
            knowledge_id="knowledge-ops",
            knowledge_base_id="kb-ops",
            text="# Ops\n\nThe latency_budget is 120ms for search.",
        )
        repo = ChunkRepository(conn)
        targets = [SearchTarget(knowledge_base_id="kb-ops", tenant_id=9)]

        grep_result = GrepChunksTool(repo, search_targets=targets).execute(
            {"query": "latency_budget"}
        )
        self.assertTrue(grep_result.success)
        self.assertEqual(grep_result.data["count"], 1)
        self.assertIn("latency_budget", grep_result.output)

        list_result = ListKnowledgeChunksTool(repo, search_targets=targets).execute(
            {"knowledge_id": "knowledge-ops"}
        )
        self.assertTrue(list_result.success)
        self.assertIn("120ms", list_result.output)

        info_result = GetDocumentInfoTool(
            repo,
            search_targets=targets,
            knowledge_titles={"knowledge-ops": "Ops"},
        ).execute({"knowledge_ids": ["knowledge-ops"]})
        self.assertTrue(info_result.success)
        self.assertEqual(info_result.data["results"][0]["title"], "Ops")

    def test_grep_scope_filter_cannot_expand_to_other_kb(self):
        conn = connect(":memory:")
        persist_text_chunks(
            conn,
            tenant_id=9,
            knowledge_id="knowledge-ops",
            knowledge_base_id="kb-ops",
            text="The latency_budget is 120ms.",
        )
        repo = ChunkRepository(conn)
        tool = GrepChunksTool(
            repo,
            search_targets=[SearchTarget(knowledge_base_id="kb-ops", tenant_id=9)],
        )

        with self.assertRaisesRegex(ToolExecutionError, "no search targets available"):
            tool.execute({"query": "latency_budget", "knowledge_base_ids": ["kb-other"]})
