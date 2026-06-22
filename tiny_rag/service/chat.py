from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from tiny_rag.agent import ChatOptions, ChatResponse, FunctionCall, LLMToolCall, Message
from tiny_rag.cancellation import CancellationToken


RESERVED_HEADER_NAMES = {"authorization", "content-type"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass(frozen=True)
class ChatConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model_name: str = ""
    timeout_seconds: float = 120.0
    max_retries: int = 2
    custom_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ChatConfig":
        provider = _env_str("LLM_PROVIDER", "openai").lower()
        if provider == "deepseek":
            default_base_url = "https://api.deepseek.com"
            default_model_name = "deepseek-v4-flash"
            default_api_key = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            default_base_url = _env_str("DEEPSEEK_BASE_URL", default_base_url)
            default_model_name = _env_str("DEEPSEEK_MODEL", default_model_name)
        else:
            default_base_url = _env_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
            default_model_name = _env_str("OPENAI_MODEL", "")
            default_api_key = os.getenv("OPENAI_API_KEY", "")
        return cls(
            api_key=_env_str("LLM_API_KEY", default_api_key),
            base_url=_env_str("LLM_BASE_URL", default_base_url),
            model_name=_env_str("LLM_MODEL_NAME", default_model_name),
            timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 120.0),
            max_retries=_env_int("LLM_MAX_RETRIES", 2),
        )


@dataclass
class OpenAIChatModel:
    config: ChatConfig

    @classmethod
    def from_env(cls) -> "OpenAIChatModel":
        return cls(ChatConfig.from_env())

    def chat(self, messages: list[Message], opts: ChatOptions) -> ChatResponse:
        if not self.config.model_name:
            raise ValueError("chat model name is required")
        body = self._build_body(messages, opts)
        response = self._post_with_retry(body, cancellation_token=opts.cancellation_token)
        if response.status_code != 200:
            body_text = response.text
            if len(body_text) > 1000:
                body_text = body_text[:1000] + "... (truncated)"
            raise RuntimeError(
                f"Chat API error: Http Status {response.status_code}, Response: {body_text}"
            )
        return parse_chat_response(response.json())

    def _build_body(self, messages: list[Message], opts: ChatOptions) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [message_to_openai(message) for message in messages],
            "temperature": opts.temperature,
        }
        if opts.tools:
            body["tools"] = opts.tools
        if opts.tool_choice:
            body["tool_choice"] = opts.tool_choice
        if opts.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = opts.parallel_tool_calls
        if opts.max_tokens > 0:
            body["max_tokens"] = opts.max_tokens
        if opts.max_completion_tokens > 0:
            body["max_completion_tokens"] = opts.max_completion_tokens
        if opts.format:
            body["response_format"] = opts.format
        return body

    def _post_with_retry(
        self,
        body: dict[str, Any],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> httpx.Response:
        base_url = self.config.base_url.rstrip("/") or "https://api.openai.com/v1"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.config.api_key,
        }
        for key, value in self.config.custom_headers.items():
            if key.lower() not in RESERVED_HEADER_NAMES:
                headers[key] = value

        timeout = self.config.timeout_seconds
        if cancellation_token is not None and cancellation_token.time_remaining() is not None:
            timeout = min(timeout, max(0.001, cancellation_token.time_remaining() or 0.001))

        last_error: Exception | None = None
        with httpx.Client(timeout=timeout) as client:
            for attempt in range(self.config.max_retries + 1):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                if attempt > 0:
                    _sleep_with_cancellation(
                        min(2 ** (attempt - 1), 10),
                        cancellation_token=cancellation_token,
                    )
                try:
                    return client.post(
                        base_url + "/chat/completions",
                        headers=headers,
                        json=body,
                    )
                except Exception as exc:
                    last_error = exc

        if last_error is not None:
            raise RuntimeError(f"send chat request: {last_error}") from last_error
        raise RuntimeError("send chat request failed")


def message_to_openai(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def parse_chat_response(payload: dict[str, Any]) -> ChatResponse:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") or {}
        tool_calls.append(
            LLMToolCall(
                id=str(item.get("id") or ""),
                type=str(item.get("type") or "function"),
                function=FunctionCall(
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or ""),
                ),
            )
        )
    return ChatResponse(
        content=str(message.get("content") or ""),
        reasoning_content=str(
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        ),
        tool_calls=tool_calls,
        finish_reason=str(choice.get("finish_reason") or "stop"),
        usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
    )


def _sleep_with_cancellation(
    seconds: float,
    *,
    cancellation_token: CancellationToken | None = None,
) -> None:
    if cancellation_token is None:
        time.sleep(seconds)
        return
    deadline = time.monotonic() + seconds
    while True:
        cancellation_token.raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))
