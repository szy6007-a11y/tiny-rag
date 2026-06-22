from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from tiny_rag.cancellation import CancellationToken
from tiny_rag.persistence.chunk_models import ChunkRow

from .definitions import ToolListKnowledgeChunks
from .grep_chunks import clean_string_array, normalize_search_targets
from .knowledge_search import SearchTarget, SearchTargets, xml_escape
from .tool import BaseTool, ToolExecutionError, ToolResult


listKnowledgeChunksDescription = """Read full chunk content from a document or FAQ hit.

Use this after grep_chunks or knowledge_search to inspect the actual evidence
before answering. For document hits, pass knowledge_id. For FAQ hits, pass
faq_id/chunk_id."""


listKnowledgeChunksSchema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "knowledge_id": {
            "type": "string",
            "description": "Document knowledge ID returned by search tools.",
        },
        "faq_id": {
            "type": "string",
            "description": "FAQ chunk ID returned by search tools.",
        },
        "knowledge_base_ids": {
            "type": "array",
            "description": "Optional KB scope filter. Cannot expand the configured runtime scope.",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 10,
        },
        "offset": {
            "type": "integer",
            "description": "Zero-based chunk offset.",
            "minimum": 0,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum chunks to return.",
            "minimum": 1,
            "maximum": 200,
        },
    },
    "required": [],
}


@dataclass(frozen=True)
class ListKnowledgeChunksInput:
    knowledge_id: str = ""
    faq_id: str = ""
    knowledge_base_ids: list[str] = field(default_factory=list)
    offset: int = 0
    limit: int = 100


class ListKnowledgeChunksTool(BaseTool):
    def __init__(
        self,
        chunk_repository: object,
        *,
        search_targets: Sequence[SearchTarget | str | dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            ToolListKnowledgeChunks,
            listKnowledgeChunksDescription,
            listKnowledgeChunksSchema,
        )
        self.chunk_repository = chunk_repository
        self.search_targets = normalize_search_targets(search_targets)

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

        if input_data.faq_id:
            chunks = self._load_faq_chunk(input_data.faq_id, targets)
        else:
            chunks = self._load_document_chunks(input_data.knowledge_id, targets)

        chunks = chunks[input_data.offset : input_data.offset + input_data.limit]
        return self._format_output(input_data, chunks)

    def _parse_args(self, args: Any) -> ListKnowledgeChunksInput:
        if not isinstance(args, dict):
            result = ToolResult(
                success=False,
                error=f"Failed to parse args: expected object, got {type(args).__name__}",
            )
            raise ToolExecutionError(result.error, result)
        knowledge_id = str(args.get("knowledge_id") or "").strip()
        faq_id = str(args.get("faq_id") or "").strip()
        if not knowledge_id and not faq_id:
            result = ToolResult(success=False, error="knowledge_id or faq_id is required")
            raise ToolExecutionError(result.error, result)
        offset = int(args.get("offset") or 0)
        limit = int(args.get("limit") or 100)
        return ListKnowledgeChunksInput(
            knowledge_id=knowledge_id,
            faq_id=faq_id,
            knowledge_base_ids=clean_string_array(args.get("knowledge_base_ids"), max_items=10),
            offset=max(0, offset),
            limit=min(200, max(1, limit)),
        )

    def _load_faq_chunk(self, chunk_id: str, targets: SearchTargets) -> list[ChunkRow]:
        for target in targets:
            chunks = self.chunk_repository.list_chunks_by_id(
                tenant_id=target.tenant_id,
                chunk_ids=[chunk_id],
            )
            chunks = [chunk for chunk in chunks if _chunk_allowed(chunk, target)]
            if chunks:
                return chunks
        return []

    def _load_document_chunks(self, knowledge_id: str, targets: SearchTargets) -> list[ChunkRow]:
        out: list[ChunkRow] = []
        seen: set[str] = set()
        for target in targets:
            if target.knowledge_ids and knowledge_id not in target.knowledge_ids:
                continue
            chunks = self.chunk_repository.list_chunks_by_knowledge_id(
                tenant_id=target.tenant_id,
                knowledge_id=knowledge_id,
            )
            for chunk in chunks:
                if chunk.id in seen or not _chunk_allowed(chunk, target):
                    continue
                seen.add(chunk.id)
                out.append(chunk)
        out.sort(key=lambda chunk: (chunk.chunk_index, chunk.seq_id or 0))
        return out

    def _format_output(
        self,
        input_data: ListKnowledgeChunksInput,
        chunks: Sequence[ChunkRow],
    ) -> ToolResult:
        if not chunks:
            return ToolResult(
                success=True,
                output="No chunks found in the configured knowledge scope.",
                data={"results": [], "count": 0},
            )

        root_attrs = []
        if input_data.knowledge_id:
            root_attrs.append(f'knowledge_id="{xml_escape(input_data.knowledge_id)}"')
        if input_data.faq_id:
            root_attrs.append(f'faq_id="{xml_escape(input_data.faq_id)}"')
        root_attrs.append(f'count="{len(chunks)}"')
        out = ["<knowledge_chunks " + " ".join(root_attrs) + ">"]
        data_results: list[dict[str, Any]] = []
        for chunk in chunks:
            out.append(
                f'<chunk chunk_id="{xml_escape(chunk.id)}" '
                f'chunk_index="{chunk.chunk_index}" '
                f'knowledge_id="{xml_escape(chunk.knowledge_id)}" '
                f'knowledge_base_id="{xml_escape(chunk.knowledge_base_id)}">'
            )
            out.append(f"<content>{xml_escape(chunk.content)}</content>")
            out.append("</chunk>")
            data_results.append(
                {
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "knowledge_id": chunk.knowledge_id,
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "content": chunk.content,
                }
            )
        out.append("</knowledge_chunks>")
        return ToolResult(
            success=True,
            output="\n".join(out),
            data={
                "results": data_results,
                "count": len(data_results),
                "display_type": "knowledge_chunks",
            },
        )


def _filter_by_kb(targets: SearchTargets, knowledge_base_ids: Sequence[str]) -> SearchTargets:
    allowed = {item for item in knowledge_base_ids if item}
    if not allowed:
        return targets
    return SearchTargets([target for target in targets if target.knowledge_base_id in allowed])


def _chunk_allowed(chunk: ChunkRow, target: SearchTarget) -> bool:
    if chunk.knowledge_base_id != target.knowledge_base_id:
        return False
    if target.knowledge_ids and chunk.knowledge_id not in target.knowledge_ids:
        return False
    return True
