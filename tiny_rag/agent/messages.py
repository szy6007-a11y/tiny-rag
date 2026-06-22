from __future__ import annotations

import json
from datetime import datetime, timezone

from .types import LLMToolCall, Message


REDACTED_RETRIEVAL_RESULT = (
    "[Previous retrieval result omitted - knowledge base may have changed. "
    "Please perform a fresh search.]"
)


def sanitize_messages(messages: list[Message]) -> list[Message]:
    if not messages:
        return messages

    result: list[Message] = []
    for index, original in enumerate(messages):
        msg = _clone_message(original)
        if msg.content == "" and msg.role not in ("system", "tool") and not msg.tool_calls:
            continue

        if result and msg.role != "tool":
            prev = result[-1]
            if prev.role == msg.role and prev.role != "tool":
                prev.content += "\n\n" + msg.content
                if msg.images:
                    prev.images.extend(msg.images)
                if msg.reasoning_content:
                    if prev.reasoning_content:
                        prev.reasoning_content += "\n\n" + msg.reasoning_content
                    else:
                        prev.reasoning_content = msg.reasoning_content
                if msg.tool_calls:
                    prev.tool_calls.extend(msg.tool_calls)
                continue

        if msg.role == "tool" and msg.tool_call_id:
            if not _has_matching_tool_call(messages[:index], msg.tool_call_id):
                msg.role = "system"
                msg.content = f"[Tool result for {msg.name}]: {msg.content}"
                msg.tool_call_id = ""
                msg.name = ""

        result.append(msg)

    return result


def redact_history_retrieval_results(
    messages: list[Message],
    *,
    retain_retrieval_history: bool,
) -> list[Message]:
    if retain_retrieval_history:
        return [_clone_message(message) for message in messages]

    redacted: list[Message] = []
    for message in messages:
        msg = _clone_message(message)
        if msg.role == "tool" and msg.name == "knowledge_search":
            msg.content = REDACTED_RETRIEVAL_RESULT
        redacted.append(msg)
    return redacted


def build_runtime_context_block(
    *,
    knowledge_base_ids: list[str],
    knowledge_ids: list[str],
) -> str:
    lines = [
        '<runtime_context note="turn metadata; current retrieval scope">',
        f"  <current_time>{datetime.now(timezone.utc).isoformat()}</current_time>",
    ]
    if knowledge_base_ids:
        lines.append("  <available_knowledge_base_ids>")
        for kb_id in knowledge_base_ids:
            lines.append(f'    <knowledge_base id="{_escape_xml_attr(kb_id)}" />')
        lines.append("  </available_knowledge_base_ids>")
    if knowledge_ids:
        lines.append('  <selected_knowledge_ids scope="authoritative_for_this_turn">')
        for knowledge_id in knowledge_ids:
            lines.append(f'    <knowledge id="{_escape_xml_attr(knowledge_id)}" />')
        lines.append("  </selected_knowledge_ids>")

    kb_scope = ",".join(knowledge_base_ids)
    knowledge_scope = ",".join(knowledge_ids)
    lines.append(
        '  <current_query_scope '
        f'knowledge_base_ids="{_escape_xml_attr(kb_scope)}" '
        f'knowledge_ids="{_escape_xml_attr(knowledge_scope)}" />'
    )
    lines.append(
        "  <answer_instruction>When you have gathered enough information, "
        "write your complete user-facing answer as plain assistant text and stop."
        "</answer_instruction>"
    )
    lines.append("</runtime_context>")
    return "\n".join(lines)


def estimate_message_tokens(message: Message) -> int:
    text = "\n".join(
        [
            message.role,
            message.content or "",
            message.name or "",
            message.tool_call_id or "",
            message.reasoning_content or "",
            json.dumps(_tool_calls_to_dicts(message.tool_calls), ensure_ascii=False),
        ]
    )
    return (len(text) + 3) // 4 + 8


def estimate_messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def compress_context(messages: list[Message], max_tokens: int, current_tokens: int) -> list[Message]:
    if max_tokens <= 0 or len(messages) <= 2 or current_tokens <= max_tokens:
        return messages

    last_user_index = len(messages) - 1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            last_user_index = index
            break

    prefix = messages[:last_user_index]
    tail = messages[last_user_index:]
    if not prefix:
        return messages

    leading_system: list[Message] = []
    history_start = 0
    while history_start < len(prefix) and prefix[history_start].role == "system":
        leading_system.append(prefix[history_start])
        history_start += 1

    groups = _group_message_units(prefix[history_start:])
    kept_groups = list(groups)
    candidate = leading_system + [msg for group in kept_groups for msg in group] + tail
    while kept_groups and estimate_messages_tokens(candidate) > max_tokens:
        kept_groups.pop(0)
        candidate = leading_system + [msg for group in kept_groups for msg in group] + tail

    return candidate


def _group_message_units(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    index = 0
    while index < len(messages):
        msg = messages[index]
        if msg.role == "assistant" and msg.tool_calls:
            group = [msg]
            expected_ids = {tool_call.id for tool_call in msg.tool_calls if tool_call.id}
            index += 1
            while index < len(messages) and messages[index].role == "tool":
                if expected_ids and messages[index].tool_call_id not in expected_ids:
                    break
                group.append(messages[index])
                index += 1
            groups.append(group)
            continue
        groups.append([msg])
        index += 1
    return groups


def _has_matching_tool_call(messages: list[Message], tool_call_id: str) -> bool:
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        for tool_call in msg.tool_calls:
            if tool_call.id == tool_call_id:
                return True
    return False


def _clone_message(message: Message) -> Message:
    return Message(
        role=message.role,
        content=message.content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=list(message.tool_calls),
        images=list(message.images),
        reasoning_content=message.reasoning_content,
    )


def _tool_calls_to_dicts(tool_calls: list[LLMToolCall]) -> list[dict[str, object]]:
    return [
        {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        for tool_call in tool_calls
    ]


def _escape_xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
