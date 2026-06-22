from __future__ import annotations

import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from tiny_rag.agent import (
    AgentConfig,
    AgentEngine,
    AgentExecutionError,
    ChatOptions,
    ChatResponse,
    FunctionCall,
    LLMToolCall,
    Message,
    REDACTED_RETRIEVAL_RESULT,
    sanitize_messages,
)
from tiny_rag.agent.tools import (
    BaseTool,
    KnowledgeSearchTool,
    ToolKnowledgeSearch,
    ToolRegistry,
    ToolResult,
)
from tiny_rag.retrieval.models import SearchResult


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def run(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        return SimpleNamespace(
            results=[
                SearchResult(
                    id="chunk-1",
                    content="Tiny RAG uses knowledge_search through the registry.",
                    knowledge_id="doc-1",
                    knowledge_base_id="kb-1",
                    knowledge_title="Agent Notes",
                    chunk_index=0,
                    score=0.9,
                )
            ]
        )


class FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], ChatOptions]] = []

    def chat(self, messages: list[Message], opts: ChatOptions) -> ChatResponse:
        self.calls.append((list(messages), opts))
        if len(self.calls) == 1:
            return ChatResponse(
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        function=FunctionCall(
                            name=ToolKnowledgeSearch,
                            arguments='{"queries":["How does Tiny RAG use agent tools?"]}',
                        ),
                    )
                ],
            )

        self.assert_tool_result_was_appended(messages)
        return ChatResponse(content="Tiny RAG calls the registry, then answers.", finish_reason="stop")

    def assert_tool_result_was_appended(self, messages: list[Message]) -> None:
        assistant = messages[-2]
        tool = messages[-1]
        assert assistant.role == "assistant"
        assert len(assistant.tool_calls) == 1
        assert assistant.tool_calls[0].id == "call-1"
        assert assistant.tool_calls[0].function.name == ToolKnowledgeSearch
        assert tool.role == "tool"
        assert tool.tool_call_id == "call-1"
        assert tool.name == ToolKnowledgeSearch
        assert "<search_results" in tool.content


class StaticTool(BaseTool):
    def __init__(
        self,
        name: str = "static_tool",
        result: ToolResult | None = None,
        *,
        sleep_seconds: float = 0,
    ) -> None:
        super().__init__(
            name,
            f"{name} description",
            {"type": "object", "properties": {}, "required": []},
        )
        self.result = result or ToolResult(success=True, output="static tool output")
        self.sleep_seconds = sleep_seconds
        self.calls = []

    def execute(self, args):
        self.calls.append(args)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return self.result


class SlowCleanupTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            "slow_cleanup_tool",
            "slow_cleanup_tool description",
            {"type": "object", "properties": {}, "required": []},
        )
        self.cleaned = False
        self.finished = False

    def execute(self, args):
        del args
        time.sleep(0.2)
        self.finished = True
        return ToolResult(success=True, output="finished late")

    def cleanup(self):
        self.cleaned = True


class SequenceChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[list[Message], ChatOptions]] = []

    def chat(self, messages: list[Message], opts: ChatOptions) -> ChatResponse:
        self.calls.append((list(messages), opts))
        if not self.responses:
            raise RuntimeError("unexpected chat call")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(messages, opts)
        return item


class AgentLoopToolIntegrationTests(TestCase):
    def test_agent_loop_passes_tool_definitions_executes_and_observes_results(self):
        pipeline = FakePipeline()
        registry = ToolRegistry()
        registry.register_tool(
            KnowledgeSearchTool(pipeline, allowed_knowledge_base_ids=["kb-1"])
        )
        chat = FakeChatModel()
        engine = AgentEngine(
            AgentConfig(max_iterations=3, temperature=0.1, max_tool_output_chars=8000),
            chat,
            registry,
        )

        state = engine.execute("Explain agent tools.", system_prompt="You are a test agent.")

        self.assertTrue(state.is_complete)
        self.assertEqual(state.final_answer, "Tiny RAG calls the registry, then answers.")
        self.assertEqual(len(state.round_steps), 2)
        self.assertEqual(state.round_steps[0].tool_calls[0].name, ToolKnowledgeSearch)
        self.assertTrue(state.round_steps[0].tool_calls[0].result.success)
        self.assertEqual(pipeline.calls[0][0], "How does Tiny RAG use agent tools?")

        first_messages, first_opts = chat.calls[0]
        self.assertEqual(first_messages[0].role, "system")
        self.assertEqual(first_messages[-1].role, "user")
        self.assertEqual(first_opts.temperature, 0.1)
        self.assertEqual(first_opts.tools[0]["type"], "function")
        self.assertEqual(
            first_opts.tools[0]["function"]["name"],
            ToolKnowledgeSearch,
        )

    def test_llm_transient_retry_success(self):
        chat = SequenceChatModel(
            [
                RuntimeError("503 server error"),
                ChatResponse(content="retry worked", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(AgentConfig(max_iterations=2), chat, ToolRegistry())

        with patch("tiny_rag.agent.engine.sleep_before_retry", return_value=None) as sleep_mock:
            state = engine.execute("Hello")

        self.assertTrue(state.is_complete)
        self.assertEqual(state.final_answer, "retry worked")
        self.assertEqual(len(chat.calls), 2)
        sleep_mock.assert_called_once_with(1.0)

    def test_first_round_llm_failure_raises_without_success_answer_or_raw_error(self):
        raw_error = "401 auth failed: secret-provider-token"
        chat = SequenceChatModel([RuntimeError(raw_error)])
        engine = AgentEngine(AgentConfig(max_iterations=1), chat, ToolRegistry())

        with self.assertRaises(AgentExecutionError) as ctx:
            engine.execute("Question")

        self.assertEqual(
            str(ctx.exception),
            "LLM call failed before any tool results were available.",
        )
        self.assertNotIn("secret-provider-token", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    def test_llm_timeout_falls_back_to_final_synthesis_from_tool_results(self):
        def slow_response(_messages, _opts):
            time.sleep(0.05)
            return ChatResponse(content="too late", finish_reason="stop")

        def synthesis_response(messages, opts):
            self.assertEqual(opts.tools, [])
            self.assertIn("Tool static_tool returned: retrieved fact", messages[-1].content)
            self.assertIn("Based on the above tool call results", messages[-1].content)
            return ChatResponse(content="Synthesized from retrieved fact.", finish_reason="stop")

        chat = SequenceChatModel(
            [
                ChatResponse(
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            id="call-1",
                            function=FunctionCall(name="static_tool", arguments="{}"),
                        )
                    ],
                ),
                slow_response,
                slow_response,
                slow_response,
                synthesis_response,
            ]
        )
        registry = ToolRegistry()
        registry.register_tool(
            StaticTool(result=ToolResult(success=True, output="retrieved fact"))
        )
        engine = AgentEngine(
            AgentConfig(max_iterations=3, llm_call_timeout=0.01),
            chat,
            registry,
        )

        with patch("tiny_rag.agent.engine.sleep_before_retry", return_value=None):
            state = engine.execute("Question")

        self.assertTrue(state.is_complete)
        self.assertEqual(state.final_answer, "Synthesized from retrieved fact.")
        self.assertEqual(len(state.round_steps), 1)
        self.assertEqual(state.round_steps[0].tool_calls[0].result.output, "retrieved fact")

    def test_timed_out_tool_cleanup_is_deferred_until_background_execution_finishes(self):
        tool = SlowCleanupTool()
        registry = ToolRegistry()
        registry.register_tool(tool)
        chat = SequenceChatModel(
            [
                ChatResponse(
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            id="slow-1",
                            function=FunctionCall(name="slow_cleanup_tool", arguments="{}"),
                        )
                    ],
                ),
                ChatResponse(content="fallback answer", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            AgentConfig(max_iterations=1, tool_call_timeout=0.01),
            chat,
            registry,
        )

        state = engine.execute("Question")

        self.assertTrue(state.is_complete)
        self.assertFalse(tool.cleaned)
        self.assertFalse(tool.finished)
        time.sleep(0.3)
        self.assertTrue(tool.finished)
        self.assertTrue(tool.cleaned)

    def test_max_iterations_synthesizes_final_answer(self):
        def synthesis_response(messages, _opts):
            self.assertIn("Tool static_tool returned: max iteration fact", messages[-1].content)
            return ChatResponse(content="Final after max iterations.", finish_reason="stop")

        chat = SequenceChatModel(
            [
                ChatResponse(
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            id="call-1",
                            function=FunctionCall(name="static_tool", arguments="{}"),
                        )
                    ],
                ),
                synthesis_response,
            ]
        )
        registry = ToolRegistry()
        registry.register_tool(
            StaticTool(result=ToolResult(success=True, output="max iteration fact"))
        )
        engine = AgentEngine(AgentConfig(max_iterations=1), chat, registry)

        state = engine.execute("Question")

        self.assertTrue(state.is_complete)
        self.assertEqual(state.final_answer, "Final after max iterations.")

    def test_history_knowledge_search_result_redaction_and_runtime_context(self):
        assistant = Message(role="assistant", content="")
        assistant.tool_calls = [
            LLMToolCall(
                id="search-1",
                function=FunctionCall(name=ToolKnowledgeSearch, arguments="{}"),
            )
        ]
        history = [
            Message(role="user", content="old question"),
            assistant,
            Message(
                role="tool",
                content="<search_results>stale full result</search_results>",
                tool_call_id="search-1",
                name=ToolKnowledgeSearch,
            ),
        ]
        chat = SequenceChatModel([ChatResponse(content="fresh answer", finish_reason="stop")])
        engine = AgentEngine(
            AgentConfig(
                max_iterations=1,
                knowledge_bases=["kb-1"],
                knowledge_ids=["doc-1"],
            ),
            chat,
            ToolRegistry(),
        )

        state = engine.execute("new question", history=history, system_prompt="system")

        self.assertTrue(state.is_complete)
        first_messages = chat.calls[0][0]
        tool_message = [message for message in first_messages if message.role == "tool"][0]
        self.assertEqual(tool_message.content, REDACTED_RETRIEVAL_RESULT)
        self.assertNotIn("stale full result", tool_message.content)
        current_user = first_messages[-1]
        self.assertIn("<runtime_context", current_user.content)
        self.assertIn('knowledge_base id="kb-1"', current_user.content)
        self.assertIn('knowledge id="doc-1"', current_user.content)

    def test_empty_stop_retries_with_plain_text_nudge(self):
        chat = SequenceChatModel(
            [
                ChatResponse(content="", finish_reason="stop"),
                ChatResponse(content="", finish_reason="stop"),
                ChatResponse(content="now complete", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(AgentConfig(max_iterations=1), chat, ToolRegistry())

        state = engine.execute("Question")

        self.assertTrue(state.is_complete)
        self.assertEqual(state.final_answer, "now complete")
        self.assertEqual(len(chat.calls), 3)
        self.assertTrue(
            chat.calls[1][0][-1].content.endswith(
                "Please provide your complete answer now as plain text."
            )
        )
        self.assertTrue(
            chat.calls[2][0][-1].content.endswith(
                "Please provide your complete answer now as plain text."
            )
        )

    def test_repeated_content_stuck_loop_breaks_complete(self):
        chat = SequenceChatModel(
            [
                ChatResponse(content="same partial", finish_reason="length"),
                ChatResponse(content="same partial", finish_reason="length"),
            ]
        )
        engine = AgentEngine(AgentConfig(max_iterations=5), chat, ToolRegistry())

        state = engine.execute("Question")

        self.assertTrue(state.is_complete)
        self.assertEqual(state.final_answer, "same partial")
        self.assertEqual(len(chat.calls), 2)

    def test_sanitize_orphan_tool_message_converts_to_system(self):
        sanitized = sanitize_messages(
            [
                Message(
                    role="tool",
                    content="orphan output",
                    tool_call_id="missing",
                    name=ToolKnowledgeSearch,
                )
            ]
        )

        self.assertEqual(len(sanitized), 1)
        self.assertEqual(sanitized[0].role, "system")
        self.assertEqual(
            sanitized[0].content,
            "[Tool result for knowledge_search]: orphan output",
        )
        self.assertEqual(sanitized[0].tool_call_id, "")
        self.assertEqual(sanitized[0].name, "")

    def test_context_management_drops_old_history_without_splitting_tool_pairs(self):
        old_assistant = Message(role="assistant", content="old thought")
        old_assistant.tool_calls = [
            LLMToolCall(
                id="old-call",
                function=FunctionCall(name="static_tool", arguments="{}"),
            )
        ]
        recent_assistant = Message(role="assistant", content="recent thought")
        recent_assistant.tool_calls = [
            LLMToolCall(
                id="recent-call",
                function=FunctionCall(name="static_tool", arguments="{}"),
            )
        ]
        messages = [
            Message(role="system", content="system"),
            Message(role="user", content="old user " + "x" * 400),
            old_assistant,
            Message(role="tool", content="old tool " + "y" * 400, tool_call_id="old-call"),
            Message(role="user", content="recent user"),
            recent_assistant,
            Message(role="tool", content="recent tool", tool_call_id="recent-call"),
            Message(role="user", content="current user"),
        ]
        engine = AgentEngine(AgentConfig(max_context_tokens=80), SequenceChatModel([]), ToolRegistry())

        managed = engine._manage_context_window(messages)

        self.assertEqual(managed[0].role, "system")
        self.assertEqual(managed[-1].content, "current user")
        remaining_ids = {message.tool_call_id for message in managed if message.role == "tool"}
        for message in managed:
            if message.role == "assistant" and message.tool_calls:
                for tool_call in message.tool_calls:
                    self.assertIn(tool_call.id, remaining_ids)
