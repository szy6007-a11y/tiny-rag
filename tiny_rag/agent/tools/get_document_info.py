from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tiny_rag.cancellation import CancellationToken
from tiny_rag.persistence.chunk_models import ChunkRow

from .definitions import ToolGetDocumentInfo
from .grep_chunks import clean_string_array, normalize_search_targets
from .knowledge_search import SearchTarget, SearchTargets, xml_escape
from .tool import BaseTool, ToolExecutionError, ToolResult


getDocumentInfoDescription = """Inspect document metadata and chunk coverage for retrieved knowledge IDs.

Use this to verify document titles, scope, chunk counts, and index ranges. Use
knowledge_search or list_knowledge_chunks when you need actual content."""


getDocumentInfoSchema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "knowledge_ids": {
            "type": "array",
            "description": "Document knowledge IDs to inspect.",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 20,
        },
        "faq_ids": {
            "type": "array",
            "description": "FAQ chunk IDs to inspect.",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 20,
        },
        "knowledge_base_ids": {
            "type": "array",
            "description": "Optional KB scope filter. Cannot expand configured runtime scope.",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 10,
        },
    },
    "required": [],
}


@dataclass(frozen=True)
class GetDocumentInfoInput:
    knowledge_ids: list[str] = field(default_factory=list)
    faq_ids: list[str] = field(default_factory=list)
    knowledge_base_ids: list[str] = field(default_factory=list)


class GetDocumentInfoTool(BaseTool):
    def __init__(
        self,
        chunk_repository: object,
        *,
        search_targets: Sequence[SearchTarget | str | dict[str, Any]] | None = None,
        knowledge_titles: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(ToolGetDocumentInfo, getDocumentInfoDescription, getDocumentInfoSchema)
        self.chunk_repository = chunk_repository
        self.search_targets = normalize_search_targets(search_targets)
        self.knowledge_titles = dict(knowledge_titles or {})

    def execute(
        self,
        args: Any,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        input_data = self._parse_args(args)
        targets = _filter_by_kb(self.search_targets, input_data.knowledge_base_ids)
        if not targets:
            result = ToolResult(
                success=False,
                error="no knowledge bases specified and no search targets configured",
            )
            raise ToolExecutionError("no search targets available", result)

        infos = []
        for knowledge_id in input_data.knowledge_ids:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            chunks = _load_document_chunks(self.chunk_repository, targets, knowledge_id)
            if chunks:
                infos.append(_document_info(knowledge_id, chunks, self.knowledge_titles))

        for faq_id in input_data.faq_ids:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            chunks = _load_faq_chunk(self.chunk_repository, targets, faq_id)
            if chunks:
                infos.append(_faq_info(faq_id, chunks[0], self.knowledge_titles))

        return self._format_output(infos)

    def _parse_args(self, args: Any) -> GetDocumentInfoInput:
        if not isinstance(args, dict):
            result = ToolResult(
                success=False,
                error=f"Failed to parse args: expected object, got {type(args).__name__}",
            )
            raise ToolExecutionError(result.error, result)
        knowledge_ids = clean_string_array(args.get("knowledge_ids"), max_items=20)
        faq_ids = clean_string_array(args.get("faq_ids"), max_items=20)
        if not knowledge_ids and not faq_ids:
            result = ToolResult(success=False, error="knowledge_ids or faq_ids is required")
            raise ToolExecutionError(result.error, result)
        return GetDocumentInfoInput(
            knowledge_ids=knowledge_ids,
            faq_ids=faq_ids,
            knowledge_base_ids=clean_string_array(args.get("knowledge_base_ids"), max_items=10),
        )

    def _format_output(self, infos: Sequence[dict[str, Any]]) -> ToolResult:
        if not infos:
            return ToolResult(
                success=True,
                output="No document info found in the configured knowledge scope.",
                data={"results": [], "count": 0},
            )

        out = [f'<document_info_results count="{len(infos)}">']
        for info in infos:
            out.append(
                f'<document knowledge_id="{xml_escape(str(info.get("knowledge_id", "")))}" '
                f'knowledge_base_id="{xml_escape(str(info.get("knowledge_base_id", "")))}" '
                f'title="{xml_escape(str(info.get("title", "")))}" '
                f'chunk_count="{int(info.get("chunk_count", 0))}" '
                f'first_chunk_index="{int(info.get("first_chunk_index", 0))}" '
                f'last_chunk_index="{int(info.get("last_chunk_index", 0))}" />'
            )
        out.append("</document_info_results>")
        return ToolResult(
            success=True,
            output="\n".join(out),
            data={
                "results": list(infos),
                "count": len(infos),
                "display_type": "document_info",
            },
        )


def _filter_by_kb(targets: SearchTargets, knowledge_base_ids: Sequence[str]) -> SearchTargets:
    allowed = {item for item in knowledge_base_ids if item}
    if not allowed:
        return targets
    return SearchTargets([target for target in targets if target.knowledge_base_id in allowed])


def _load_document_chunks(
    chunk_repository: object,
    targets: Sequence[SearchTarget],
    knowledge_id: str,
) -> list[ChunkRow]:
    out: list[ChunkRow] = []
    seen: set[str] = set()
    for target in targets:
        if target.knowledge_ids and knowledge_id not in target.knowledge_ids:
            continue
        chunks = chunk_repository.list_chunks_by_knowledge_id(
            tenant_id=target.tenant_id,
            knowledge_id=knowledge_id,
        )
        for chunk in chunks:
            if chunk.id in seen:
                continue
            if chunk.knowledge_base_id != target.knowledge_base_id:
                continue
            seen.add(chunk.id)
            out.append(chunk)
    out.sort(key=lambda chunk: (chunk.chunk_index, chunk.seq_id or 0))
    return out


def _load_faq_chunk(
    chunk_repository: object,
    targets: Sequence[SearchTarget],
    faq_id: str,
) -> list[ChunkRow]:
    for target in targets:
        chunks = chunk_repository.list_chunks_by_id(
            tenant_id=target.tenant_id,
            chunk_ids=[faq_id],
        )
        chunks = [chunk for chunk in chunks if chunk.knowledge_base_id == target.knowledge_base_id]
        if chunks:
            return chunks
    return []


def _document_info(
    knowledge_id: str,
    chunks: Sequence[ChunkRow],
    titles: Mapping[str, str],
) -> dict[str, Any]:
    indexes = [chunk.chunk_index for chunk in chunks]
    return {
        "knowledge_id": knowledge_id,
        "knowledge_base_id": chunks[0].knowledge_base_id,
        "title": titles.get(knowledge_id, knowledge_id),
        "chunk_count": len(chunks),
        "first_chunk_index": min(indexes),
        "last_chunk_index": max(indexes),
    }


def _faq_info(
    faq_id: str,
    chunk: ChunkRow,
    titles: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "faq_id": faq_id,
        "knowledge_id": chunk.knowledge_id,
        "knowledge_base_id": chunk.knowledge_base_id,
        "title": titles.get(chunk.knowledge_id, chunk.knowledge_id),
        "chunk_count": 1,
        "first_chunk_index": chunk.chunk_index,
        "last_chunk_index": chunk.chunk_index,
    }
