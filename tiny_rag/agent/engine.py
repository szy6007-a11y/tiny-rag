from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import replace
from datetime import datetime, timezone

from tiny_rag.cancellation import CancellationToken

from .act import append_tool_results, execute_tool_calls
from .messages import (
    build_runtime_context_block,
    compress_context,
    estimate_messages_tokens,
    redact_history_retrieval_results,
    sanitize_messages,
)
from .observe import analyze_response
from .tools.registry import ToolRegistry
from .types import AgentConfig, AgentState, AgentStep, ChatModel, ChatOptions, Message


DefaultLLMCallTimeout = 120
MaxLLMRetries = 2
MaxEmptyResponseRetries = 2
MaxRepeatedResponseRounds = 2

transientErrorMarkers = [
    "429",
    "rate limit",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "server error",
    "temporarily unavailable",
]


class AgentExecutionError(RuntimeError):
    pass


class AgentEngine:
    def __init__(
        self,
        config: AgentConfig,
        chat_model: ChatModel,
        tool_registry: ToolRegistry,
    ) -> None:
        self.config = config
        self.chat_model = chat_model
        self.tool_registry = tool_registry
        self.tool_registry.set_max_tool_output_size(config.max_tool_output_chars)

    def build_tools_for_llm(self) -> list[dict]:
        return [
            definition.to_chat_tool()
            for definition in self.tool_registry.get_function_definitions()
        ]

    def execute(
        self,
        query: str,
        *,
        history: list[Message] | None = None,
        system_prompt: str = "",
    ) -> AgentState:
        state = AgentState()
        messages = self._build_initial_messages(
            query,
            history=history or [],
            system_prompt=system_prompt or self.config.system_prompt,
        )
        tools = self.build_tools_for_llm()

        try:
            empty_retries = 0
            same_content_count = 0
            last_response_content = ""
            while state.current_round < self.config.max_iterations:
                messages = self._manage_context_window(messages)
                response = self._call_llm_with_retry(
                    messages,
                    ChatOptions(
                        temperature=self.config.temperature,
                        tools=tools,
                        thinking=self.config.thinking,
                        parallel_tool_calls=self.config.parallel_tool_calls,
                    ),
                    state=state,
                    query=query,
                    system_prompt=system_prompt or self.config.system_prompt,
                )
                if response is None:
                    break

                if len(response.tool_calls) == 0 and response.content:
                    if response.content == last_response_content:
                        same_content_count += 1
                    else:
                        same_content_count = 1
                    last_response_content = response.content
                    if same_content_count >= MaxRepeatedResponseRounds:
                        state.final_answer = response.content
                        state.is_complete = True
                        break
                else:
                    same_content_count = 0
                    last_response_content = ""

                step = AgentStep(
                    iteration=state.current_round,
                    thought=response.content,
                    reasoning_content=response.reasoning_content,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                verdict = analyze_response(response, step)
                if verdict.is_done:
                    if verdict.empty_content:
                        empty_retries += 1
                        if empty_retries <= MaxEmptyResponseRetries:
                            messages.append(
                                Message(
                                    role="user",
                                    content="Please provide your complete answer now as plain text.",
                                )
                            )
                            continue
                        state.final_answer = (
                            "I'm sorry, I was unable to generate a response. Please try again."
                        )
                        state.is_complete = True
                        state.round_steps.append(verdict.step or step)
                        break

                    state.final_answer = verdict.final_answer
                    state.is_complete = True
                    state.round_steps.append(verdict.step or step)
                    break

                execute_tool_calls(
                    self.tool_registry,
                    response,
                    step,
                    parallel_tool_calls=self.config.parallel_tool_calls,
                    tool_call_timeout=self._get_tool_call_timeout(),
                )
                state.round_steps.append(step)
                messages = append_tool_results(messages, step)
                state.current_round += 1

            if not state.is_complete:
                self._handle_max_iterations(
                    query,
                    state,
                    system_prompt=system_prompt or self.config.system_prompt,
                )
            return state
        finally:
            self.tool_registry.cleanup()

    def _build_initial_messages(
        self,
        query: str,
        *,
        history: list[Message],
        system_prompt: str,
    ) -> list[Message]:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        history_messages = redact_history_retrieval_results(
            [message for message in history if message.role != "system"],
            retain_retrieval_history=self.config.retain_retrieval_history,
        )
        messages.extend(history_messages)
        runtime_context = build_runtime_context_block(
            knowledge_base_ids=list(self.config.knowledge_bases),
            knowledge_ids=list(self.config.knowledge_ids),
        )
        messages.append(Message(role="user", content=runtime_context + "\n\n" + query))
        return messages

    def _call_llm_with_retry(
        self,
        messages: list[Message],
        opts: ChatOptions,
        *,
        state: AgentState,
        query: str,
        system_prompt: str,
    ):
        last_error: Exception | None = None
        sanitized = sanitize_messages(messages)
        for attempt in range(MaxLLMRetries + 1):
            try:
                return self._call_chat_with_timeout(sanitized, opts)
            except Exception as exc:
                last_error = exc
                if not is_transient_error(exc) or attempt == MaxLLMRetries:
                    break
                sleep_before_retry(float(attempt + 1))
                sanitized = sanitize_messages(messages)

        if count_total_tool_calls(state.round_steps) > 0:
            if self._synthesize_final_answer(query, state, system_prompt=system_prompt):
                state.is_complete = True
                return None
            state.final_answer = (
                "Sorry, I was unable to generate a complete answer from the retrieved results."
            )
            state.is_complete = True
            return None

        raise AgentExecutionError(
            "LLM call failed before any tool results were available."
        ) from last_error

    def _call_chat_with_timeout(self, messages: list[Message], opts: ChatOptions):
        timeout = self._get_llm_call_timeout()
        if timeout <= 0:
            return self.chat_model.chat(messages, opts)

        cancellation_token = CancellationToken.with_timeout(timeout)
        opts = replace(opts, cancellation_token=cancellation_token)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.chat_model.chat, messages, opts)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            cancellation_token.cancel(f"LLM call timed out after {timeout:g}s")
            future.cancel()
            raise TimeoutError(f"LLM call timed out after {timeout:g}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _synthesize_final_answer(
        self,
        query: str,
        state: AgentState,
        *,
        system_prompt: str,
    ) -> bool:
        messages = self._build_final_answer_messages(query, state, system_prompt=system_prompt)
        opts = ChatOptions(temperature=self.config.temperature, thinking=False)
        sanitized = sanitize_messages(messages)
        for attempt in range(MaxLLMRetries + 1):
            try:
                response = self._call_chat_with_timeout(sanitized, opts)
                answer = analyze_final_answer(response.content)
                if not answer:
                    raise ValueError("final answer synthesis returned empty content")
                state.final_answer = answer
                return True
            except Exception as exc:
                if not is_transient_error(exc) or attempt == MaxLLMRetries:
                    break
                sleep_before_retry(float(attempt + 1))
                sanitized = sanitize_messages(messages)
        return False

    def _build_final_answer_messages(
        self,
        query: str,
        state: AgentState,
        *,
        system_prompt: str,
    ) -> list[Message]:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=query))

        for step in state.round_steps:
            for tool_call in step.tool_calls:
                output = tool_call.result.output
                messages.append(
                    Message(
                        role="user",
                        content=f"Tool {tool_call.name} returned: {output}",
                    )
                )

        final_prompt = f"""Based on the above tool call results, generate a complete answer for the user's question.

User question: {query}

Requirements:
1. Answer based on the actually retrieved content
2. Clearly cite information sources (chunk_id, document name)
3. Organize the answer in a structured format
4. If information is insufficient, honestly state so
5. IMPORTANT: Respond in the same language as the user's question

Now generate the final answer:"""
        messages.append(Message(role="user", content=final_prompt))
        return messages

    def _handle_max_iterations(
        self,
        query: str,
        state: AgentState,
        *,
        system_prompt: str,
    ) -> None:
        if not self._synthesize_final_answer(query, state, system_prompt=system_prompt):
            state.final_answer = "Sorry, I was unable to generate a complete answer."
        state.is_complete = True

    def _manage_context_window(self, messages: list[Message]) -> list[Message]:
        max_tokens = self.config.max_context_tokens
        if max_tokens <= 0:
            return messages
        current_tokens = estimate_messages_tokens(messages)
        return compress_context(messages, max_tokens, current_tokens)

    def _get_llm_call_timeout(self) -> float:
        if self.config.llm_call_timeout > 0:
            return float(self.config.llm_call_timeout)
        return float(DefaultLLMCallTimeout)

    def _get_tool_call_timeout(self) -> float:
        default_timeout = 60
        tool_call_timeout = getattr(self.config, "tool_call_timeout", default_timeout)
        tool_exec_timeout = getattr(self.config, "tool_exec_timeout", default_timeout)
        if tool_call_timeout > 0 and tool_call_timeout != default_timeout:
            return float(tool_call_timeout)
        if tool_exec_timeout > 0 and tool_exec_timeout != default_timeout:
            return float(tool_exec_timeout)
        if tool_call_timeout > 0:
            return float(tool_call_timeout)
        if tool_exec_timeout > 0:
            return float(tool_exec_timeout)
        return float(default_timeout)


def count_total_tool_calls(steps: list[AgentStep]) -> int:
    return sum(len(step.tool_calls) for step in steps)


def is_transient_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    text = str(exc).lower()
    return any(marker in text for marker in transientErrorMarkers)


def analyze_final_answer(content: str) -> str:
    from .observe import strip_think_blocks

    return strip_think_blocks(content)


def sleep_before_retry(seconds: float) -> None:
    time.sleep(seconds)
