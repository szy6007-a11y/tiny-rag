from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from tiny_rag.cancellation import CancellationToken

from .tools.json_repair import repair_json
from .tools.registry import ToolRegistry, toolErrorHint
from .tools.tool import ToolResult
from .types import AgentStep, ChatResponse, LLMToolCall, Message, ToolCall


DefaultToolCallTimeout = 60


def normalize_tool_call_id(tool_call_id: str, tool_name: str, index: int) -> str:
    if tool_call_id:
        return tool_call_id
    return f"{tool_name}-{index}"


def run_tool_call(
    registry: ToolRegistry,
    llm_tool_call: LLMToolCall,
    index: int,
    *,
    tool_call_timeout: float = DefaultToolCallTimeout,
) -> ToolCall:
    tool_call_id = normalize_tool_call_id(
        llm_tool_call.id,
        llm_tool_call.function.name,
        index,
    )
    args_str = llm_tool_call.function.arguments or "{}"

    try:
        try:
            parsed = json.loads(args_str)
        except Exception:
            repaired = repair_json(args_str)
            parsed = json.loads(repaired)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        args: dict[str, Any] = parsed
    except Exception as exc:
        return ToolCall(
            id=tool_call_id,
            name=llm_tool_call.function.name,
            args={"_raw": args_str},
            provider_metadata=llm_tool_call.provider_metadata,
            result=ToolResult(
                success=False,
                error=f"Failed to parse tool arguments: {exc}{toolErrorHint}",
            ),
        )

    started = time.monotonic()
    result = _execute_tool_with_timeout(
        registry,
        llm_tool_call.function.name,
        args,
        timeout_seconds=tool_call_timeout,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    return ToolCall(
        id=tool_call_id,
        name=llm_tool_call.function.name,
        args=args,
        result=result,
        duration=duration_ms,
        provider_metadata=llm_tool_call.provider_metadata,
    )


def execute_tool_calls(
    registry: ToolRegistry,
    response: ChatResponse,
    step: AgentStep,
    *,
    parallel_tool_calls: bool = False,
    tool_call_timeout: float = DefaultToolCallTimeout,
) -> None:
    calls = list(response.tool_calls or [])
    if not calls:
        return

    if parallel_tool_calls and len(calls) >= 2:
        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            results = list(
                executor.map(
                    lambda pair: run_tool_call(
                        registry,
                        pair[1],
                        pair[0],
                        tool_call_timeout=tool_call_timeout,
                    ),
                    enumerate(calls),
                )
            )
        step.tool_calls.extend(results)
        return

    for index, tool_call in enumerate(calls):
        step.tool_calls.append(
            run_tool_call(
                registry,
                tool_call,
                index,
                tool_call_timeout=tool_call_timeout,
            )
        )


def append_tool_results(messages: list[Message], step: AgentStep) -> list[Message]:
    if step.thought or step.tool_calls or step.reasoning_content:
        assistant_msg = Message(
            role="assistant",
            content=step.thought,
            reasoning_content=step.reasoning_content,
        )
        if step.tool_calls:
            assistant_msg.tool_calls = [
                LLMToolCall(
                    id=tool_call.id,
                    type="function",
                    provider_metadata=tool_call.provider_metadata,
                    function=tool_call_to_function(tool_call),
                )
                for tool_call in step.tool_calls
            ]
        messages.append(assistant_msg)

    for tool_call in step.tool_calls:
        content = (
            tool_call.result.output
            if tool_call.result.success
            else f"Error: {tool_call.result.error}"
        )
        messages.append(
            Message(
                role="tool",
                content=content,
                tool_call_id=tool_call.id,
                name=tool_call.name,
            )
        )
    return messages


def tool_call_to_function(tool_call: ToolCall):
    from .types import FunctionCall

    return FunctionCall(
        name=tool_call.name,
        arguments=json.dumps(tool_call.args, ensure_ascii=False, separators=(",", ":")),
    )


def _execute_tool_with_timeout(
    registry: ToolRegistry,
    name: str,
    args: dict[str, Any],
    *,
    timeout_seconds: float,
) -> ToolResult:
    if timeout_seconds <= 0:
        return registry.execute_tool(name, args)

    cancellation_token = CancellationToken.with_timeout(timeout_seconds)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        registry.execute_tool,
        name,
        args,
        cancellation_token=cancellation_token,
    )
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        cancellation_token.cancel(f"Tool execution timed out after {timeout_seconds:g}s")
        future.cancel()
        if not future.done():
            registry.track_background_tool_execution(name, future)
        return ToolResult(
            success=False,
            error=(
                f"Tool execution timed out after {timeout_seconds:g}s"
                f"{toolErrorHint}"
            ),
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
