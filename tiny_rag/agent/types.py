from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tiny_rag.cancellation import CancellationToken

from .tools.tool import ToolResult


DefaultMaxContextTokens = 200000


@dataclass
class AgentConfig:
    max_iterations: int = 20
    allowed_tools: list[str] = field(default_factory=list)
    temperature: float = 0.7
    knowledge_bases: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    system_prompt: str = ""
    use_custom_system_prompt: bool = False
    multi_turn_enabled: bool = True
    history_turns: int = 20
    thinking: bool | None = None
    retrieve_kb_only_when_mentioned: bool = False
    retain_retrieval_history: bool = False
    skills_enabled: bool = False
    skill_dirs: list[str] = field(default_factory=list)
    allowed_skills: list[str] = field(default_factory=list)
    llm_call_timeout: int = 120
    tool_call_timeout: int = 60
    tool_exec_timeout: int = 60
    max_tool_output_chars: int = 16000
    max_context_tokens: int = DefaultMaxContextTokens
    parallel_tool_calls: bool = False


@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class LLMToolCall:
    id: str
    function: FunctionCall
    type: str = "function"
    provider_metadata: dict[str, Any] | None = None


@dataclass
class Message:
    role: str
    content: str
    name: str = ""
    tool_call_id: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    reasoning_content: str = ""


@dataclass
class ChatOptions:
    temperature: float = 0.7
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str = ""
    parallel_tool_calls: bool | None = None
    thinking: bool | None = None
    max_tokens: int = 0
    max_completion_tokens: int = 0
    format: dict[str, Any] | None = None
    cancellation_token: CancellationToken | None = None


@dataclass
class ChatResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None
    answer_streamed: bool = False
    answer_event_id: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    result: ToolResult
    duration: int = 0
    reflection: str = ""
    provider_metadata: dict[str, Any] | None = None


@dataclass
class AgentStep:
    iteration: int
    thought: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: str = ""

    def get_observations(self) -> list[str]:
        observations: list[str] = []
        for tool_call in self.tool_calls:
            if tool_call.result.output:
                observations.append(tool_call.result.output)
            if tool_call.reflection:
                observations.append("Reflection: " + tool_call.reflection)
        return observations


@dataclass
class AgentState:
    current_round: int = 0
    round_steps: list[AgentStep] = field(default_factory=list)
    is_complete: bool = False
    final_answer: str = ""
    knowledge_refs: list[dict[str, Any]] = field(default_factory=list)


class ChatModel(Protocol):
    def chat(self, messages: list[Message], opts: ChatOptions) -> ChatResponse:
        ...
