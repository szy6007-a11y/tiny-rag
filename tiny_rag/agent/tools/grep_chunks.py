from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from tiny_rag.cancellation import CancellationToken
from tiny_rag.persistence.chunk_models import ChunkRow

from .definitions import ToolGrepChunks
from .knowledge_search import SearchTarget, SearchTargets, xml_escape
from .tool import BaseTool, ToolExecutionError, ToolResult


grepChunksDescription = """Keyword/regex search tool for locating exact terms in knowledge base chunks.

Use this when the user asks about literal names, identifiers, error codes, values,
or strings that should be matched exactly before semantic expansion. Results are
snippets for locating relevant documents; call list_knowledge_chunks to read the
full chunk content before answering."""


grepChunksSchema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "REQUIRED: case-insensitive regex or literal keyword expression.",
            "minLength": 1,
        },
        "knowledge_base_ids": {
            "type": "array",
            "description": "Optional KB scope filter. Cannot expand the configured runtime scope.",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 10,
        },
        "knowledge_ids": {
            "type": "array",
            "description": "Optional document IDs to search within the configured runtime scope.",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 20,
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum snippets to return.",
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": ["query"],
}


@dataclass(frozen=True)
class GrepChunksInput:
    query: str
    knowledge_base_ids: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    top_k: int = 10


class GrepChunksTool(BaseTool):
    def __init__(
        self,
        chunk_repository: object,
        *,
        search_targets: Sequence[SearchTarget | str | dict[str, Any]] | None = None,
        top_k: int = 10,
    ) -> None:
        super().__init__(ToolGrepChunks, grepChunksDescription, grepChunksSchema)
        self.chunk_repository = chunk_repository
        self.search_targets = normalize_search_targets(search_targets)
        self.top_k = top_k

    def execute(
        self,
        args: Any,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        input_data = self._parse_args(args)
        targets = filter_search_targets(
            self.search_targets,
            knowledge_base_ids=input_data.knowledge_base_ids,
            knowledge_ids=input_data.knowledge_ids,
        )
        if not targets:
            result = ToolResult(
                success=False,
                error="no knowledge bases specified and no search targets configured",
            )
            raise ToolExecutionError("no search targets available", result)

        try:
            pattern = re.compile(input_data.query, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            result = ToolResult(success=False, error=f"invalid grep regex: {exc}")
            raise ToolExecutionError(result.error, result) from exc

        hits: list[tuple[ChunkRow, str]] = []
        seen: set[str] = set()
        for chunk in iter_target_chunks(self.chunk_repository, targets):
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            if chunk.id in seen:
                continue
            match = pattern.search(chunk.content or "")
            if match is None:
                continue
            seen.add(chunk.id)
            hits.append((chunk, snippet_around_match(chunk.content, match)))
            if len(hits) >= input_data.top_k:
                break

        return self._format_output(input_data, targets, hits)

    def _parse_args(self, args: Any) -> GrepChunksInput:
        if not isinstance(args, dict):
            result = ToolResult(
                success=False,
                error=f"Failed to parse args: expected object, got {type(args).__name__}",
            )
            raise ToolExecutionError(result.error, result)
        query = str(args.get("query") or "").strip()
        if not query:
            result = ToolResult(success=False, error="query parameter is required")
            raise ToolExecutionError(result.error, result)
        top_k = int(args.get("top_k") or self.top_k or 10)
        top_k = min(50, max(1, top_k))
        return GrepChunksInput(
            query=query,
            knowledge_base_ids=clean_string_array(args.get("knowledge_base_ids"), max_items=10),
            knowledge_ids=clean_string_array(args.get("knowledge_ids"), max_items=20),
            top_k=top_k,
        )

    def _format_output(
        self,
        input_data: GrepChunksInput,
        targets: SearchTargets,
        hits: Sequence[tuple[ChunkRow, str]],
    ) -> ToolResult:
        kb_ids = targets.get_all_knowledge_base_ids()
        if not hits:
            return ToolResult(
                success=True,
                output=(
                    f"No literal matches found for {input_data.query!r} in "
                    f"{len(kb_ids)} knowledge base(s)."
                ),
                data={
                    "knowledge_base_ids": kb_ids,
                    "query": input_data.query,
                    "results": [],
                    "count": 0,
                },
            )

        out = [f'<grep_results count="{len(hits)}" query="{xml_escape(input_data.query)}">']
        data_results: list[dict[str, Any]] = []
        for index, (chunk, snippet) in enumerate(hits, start=1):
            out.append(
                f'<chunk rank="{index}" chunk_id="{xml_escape(chunk.id)}" '
                f'chunk_index="{chunk.chunk_index}" '
                f'knowledge_id="{xml_escape(chunk.knowledge_id)}" '
                f'knowledge_base_id="{xml_escape(chunk.knowledge_base_id)}">'
            )
            out.append(f"<match_snippet>{xml_escape(snippet)}</match_snippet>")
            out.append("</chunk>")
            data_results.append(
                {
                    "result_index": index,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "knowledge_id": chunk.knowledge_id,
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "match_snippet": snippet,
                }
            )
        out.append("</grep_results>")
        return ToolResult(
            success=True,
            output="\n".join(out),
            data={
                "knowledge_base_ids": kb_ids,
                "query": input_data.query,
                "results": data_results,
                "count": len(data_results),
                "display_type": "grep_results",
            },
        )


def normalize_search_targets(
    search_targets: Sequence[SearchTarget | str | dict[str, Any]] | None,
) -> SearchTargets:
    targets = SearchTargets()
    for raw in list(search_targets or []):
        target = coerce_search_target(raw)
        if target is not None and target.knowledge_base_id:
            targets.append(target)
    return targets


def coerce_search_target(raw: SearchTarget | str | dict[str, Any]) -> SearchTarget | None:
    if isinstance(raw, SearchTarget):
        return raw
    if isinstance(raw, str):
        return SearchTarget(knowledge_base_id=raw)
    if isinstance(raw, dict):
        kb_id = str(
            raw.get("knowledge_base_id")
            or raw.get("KnowledgeBaseID")
            or raw.get("id")
            or ""
        )
        knowledge_ids_raw = raw.get("knowledge_ids") or raw.get("KnowledgeIDs") or []
        if isinstance(knowledge_ids_raw, str):
            knowledge_ids = (knowledge_ids_raw,)
        else:
            knowledge_ids = tuple(str(item) for item in knowledge_ids_raw)
        return SearchTarget(
            knowledge_base_id=kb_id,
            knowledge_ids=knowledge_ids,
            tenant_id=int(raw.get("tenant_id") or raw.get("TenantID") or 0),
            target_type=str(raw.get("type") or raw.get("Type") or "knowledge_base"),
            knowledge_base_type=str(
                raw.get("knowledge_base_type") or raw.get("KnowledgeBaseType") or ""
            ),
        )
    return None


def filter_search_targets(
    targets: SearchTargets,
    *,
    knowledge_base_ids: Sequence[str] | None = None,
    knowledge_ids: Sequence[str] | None = None,
) -> SearchTargets:
    allowed_kbs = {item for item in (knowledge_base_ids or []) if item}
    allowed_knowledge = {item for item in (knowledge_ids or []) if item}
    out = SearchTargets()
    for target in targets:
        if allowed_kbs and target.knowledge_base_id not in allowed_kbs:
            continue
        if not allowed_knowledge:
            out.append(target)
            continue
        if target.knowledge_ids:
            scoped = tuple(item for item in target.knowledge_ids if item in allowed_knowledge)
            if scoped:
                out.append(
                    SearchTarget(
                        knowledge_base_id=target.knowledge_base_id,
                        knowledge_ids=scoped,
                        tenant_id=target.tenant_id,
                        target_type=target.target_type,
                        knowledge_base_type=target.knowledge_base_type,
                    )
                )
            continue
        out.append(
            SearchTarget(
                knowledge_base_id=target.knowledge_base_id,
                knowledge_ids=tuple(sorted(allowed_knowledge)),
                tenant_id=target.tenant_id,
                target_type=target.target_type,
                knowledge_base_type=target.knowledge_base_type,
            )
        )
    return out


def iter_target_chunks(chunk_repository: object, targets: Sequence[SearchTarget]):
    for target in targets:
        if target.knowledge_ids:
            for knowledge_id in target.knowledge_ids:
                yield from chunk_repository.list_chunks_by_knowledge_id(
                    tenant_id=target.tenant_id,
                    knowledge_id=knowledge_id,
                )
        else:
            yield from chunk_repository.list_chunks_by_knowledge_base_id(
                tenant_id=target.tenant_id,
                knowledge_base_id=target.knowledge_base_id,
            )


def clean_string_array(raw: Any, *, max_items: int) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        result = ToolResult(success=False, error="array parameter must be an array")
        raise ToolExecutionError(result.error, result)
    if len(raw) > max_items:
        result = ToolResult(
            success=False,
            error=f"array parameter must contain at most {max_items} items",
        )
        raise ToolExecutionError(result.error, result)
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def snippet_around_match(content: str, match: re.Match[str], context_chars: int = 160) -> str:
    start = max(0, match.start() - context_chars)
    end = min(len(content), match.end() + context_chars)
    snippet = content[start:end].replace("\n", " ")
    while "  " in snippet:
        snippet = snippet.replace("  ", " ")
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(content) else ""
    return prefix + snippet.strip() + suffix
