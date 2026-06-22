from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tiny_rag.cancellation import CancellationToken
from tiny_rag.retrieval.models import (
    MATCH_TYPE_GRAPH,
    MATCH_TYPE_HISTORY,
    MATCH_TYPE_NEAR_BY_CHUNK,
    MATCH_TYPE_PARENT_CHUNK,
    MATCH_TYPE_RELATION_CHUNK,
)
from tiny_rag.indexing.models import MATCH_TYPE_EMBEDDING, MATCH_TYPE_KEYWORDS


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    data: dict[str, Any] | None = None
    error: str = ""
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": self.success,
            "output": self.output,
        }
        if self.data is not None:
            payload["data"] = self.data
        if self.error:
            payload["error"] = self.error
        if self.images:
            payload["images"] = list(self.images)
        return payload


@dataclass(frozen=True)
class FunctionDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_chat_tool(self) -> dict[str, Any]:
        return {"type": "function", "function": self.to_dict()}


class ToolExecutionError(Exception):
    def __init__(self, message: str, result: ToolResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class Tool(Protocol):
    def name(self) -> str:
        ...

    def description(self) -> str:
        ...

    def parameters(self) -> dict[str, Any]:
        ...

    def execute(
        self,
        args: Any,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        ...


class Cleanable(Protocol):
    def cleanup(self) -> None:
        ...


class BaseTool:
    def __init__(self, name: str, description: str, schema: dict[str, Any]) -> None:
        self._name = name
        self._description = description
        self._schema = schema

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def parameters(self) -> dict[str, Any]:
        return self._schema


def get_relevance_level(score: float) -> str:
    if score >= 0.8:
        return "High Relevance"
    if score >= 0.6:
        return "Medium Relevance"
    if score >= 0.4:
        return "Low Relevance"
    return "Weak Relevance"


def format_match_type(match_type: int) -> str:
    if match_type == MATCH_TYPE_EMBEDDING:
        return "Vector Match"
    if match_type == MATCH_TYPE_KEYWORDS:
        return "Keyword Match"
    if match_type == MATCH_TYPE_NEAR_BY_CHUNK:
        return "Adjacent Chunk Match"
    if match_type == MATCH_TYPE_HISTORY:
        return "History Match"
    if match_type == MATCH_TYPE_PARENT_CHUNK:
        return "Parent Chunk Match"
    if match_type == MATCH_TYPE_RELATION_CHUNK:
        return "Relation Chunk Match"
    if match_type == MATCH_TYPE_GRAPH:
        return "Graph Match"
    return f"Unknown Type({match_type})"
