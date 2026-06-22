from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterable, Sequence

from tiny_rag.cancellation import CancellationToken, OperationCancelled
from tiny_rag.retrieval import SearchPipeline
from tiny_rag.retrieval.models import CHUNK_TYPE_FAQ, KNOWLEDGE_BASE_TYPE_FAQ, SearchResult
from tiny_rag.retrieval.utils import build_content_signature

from .definitions import ToolKnowledgeSearch
from .tool import BaseTool, ToolExecutionError, ToolResult


knowledgeSearchDescription = """Semantic/vector search tool for retrieving knowledge by meaning, intent, and conceptual relevance.

This tool uses embeddings to understand the user's query and find semantically similar content across knowledge base chunks.

## Purpose
Designed for high-level understanding tasks, such as:
- conceptual explanations
- topic overviews
- reasoning-based information needs
- contextual or intent-driven retrieval
- queries that cannot be answered with literal keyword matching

The tool searches by MEANING rather than exact text. It identifies chunks that are conceptually relevant even when the wording differs.

## What the Tool Does NOT Do
- Does NOT perform exact keyword matching
- Does NOT search for specific named entities
- Should NOT be used for literal lookup tasks
- Should NOT receive long raw text or user messages as queries
- Should NOT be used to locate specific strings or error codes

For literal/keyword/entity search, another tool should be used.

## Required Input Behavior
"queries" must contain **1-5 short, well-formed semantic questions or conceptual statements** that clearly express the meaning the model is trying to retrieve.

Each query should represent a **concept, idea, topic, explanation, or intent**, such as:
- abstract topics
- definitions
- mechanisms
- best practices
- comparisons
- how/why questions

Avoid:
- keyword lists
- raw text from user messages
- full paragraphs
- unprocessed input

## Examples of valid query shapes (not content):
- "What is the main idea of..."
- "How does X work in general?"
- "Explain the purpose of..."
- "What are the key principles behind..."
- "Overview of ..."

## Parameters
- queries (required): 1-5 semantic questions or conceptual statements.
  These should reflect the meaning or topic you want embeddings to capture.
- knowledge_base_ids (optional): limit the search scope.

## Output
Returns chunks ranked by semantic similarity, reranked when applicable.
Results represent conceptual relevance, not literal keyword overlap."""


knowledgeSearchSchema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "description": "REQUIRED: 1-5 semantic questions/topics (e.g., ['What is RAG?', 'RAG benefits'])",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "knowledge_base_ids": {
            "type": "array",
            "description": "Optional: KB IDs to search",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 10,
        },
    },
    "required": ["queries"],
}


@dataclass(frozen=True)
class KnowledgeSearchInput:
    queries: list[str] = field(default_factory=list)
    knowledge_base_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchTarget:
    knowledge_base_id: str
    knowledge_ids: tuple[str, ...] = ()
    tenant_id: int = 0
    target_type: str = "knowledge_base"
    knowledge_base_type: str = ""


class SearchTargets(list[SearchTarget]):
    def get_all_knowledge_base_ids(self) -> list[str]:
        return _unique([target.knowledge_base_id for target in self if target.knowledge_base_id])

    def get_tenant_id_for_kb(self, knowledge_base_id: str) -> int:
        for target in self:
            if target.knowledge_base_id == knowledge_base_id:
                return target.tenant_id
        return 0

    def get_type_for_kb(self, knowledge_base_id: str) -> str:
        for target in self:
            if target.knowledge_base_id == knowledge_base_id:
                return target.knowledge_base_type
        return ""


@dataclass
class _SearchResultWithMeta:
    result: SearchResult
    source_query: str
    query_type: str = "hybrid"
    knowledge_base_id: str = ""
    knowledge_base_type: str = ""


@dataclass(frozen=True)
class _SearchBranchFailure:
    query: str
    knowledge_base_ids: tuple[str, ...]
    knowledge_ids: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "knowledge_base_ids": list(self.knowledge_base_ids),
            "knowledge_ids": list(self.knowledge_ids),
            "error": self.error,
        }


class KnowledgeSearchTool(BaseTool):
    def __init__(
        self,
        search_pipeline: SearchPipeline,
        *,
        search_targets: Sequence[SearchTarget | str | dict[str, Any]] | None = None,
        allowed_knowledge_base_ids: Sequence[str] | None = None,
        knowledge_base_types: dict[str, str] | None = None,
        top_k: int = 5,
        vector_threshold: float = 0.6,
        keyword_threshold: float = 0.5,
        rerank_threshold: float = 0.3,
    ) -> None:
        super().__init__(ToolKnowledgeSearch, knowledgeSearchDescription, knowledgeSearchSchema)
        self.search_pipeline = search_pipeline
        self.search_targets = self._normalize_search_targets(
            search_targets=search_targets,
            allowed_knowledge_base_ids=allowed_knowledge_base_ids,
            knowledge_base_types=knowledge_base_types or {},
        )
        self.top_k = top_k
        self.vector_threshold = vector_threshold
        self.keyword_threshold = keyword_threshold
        self.rerank_threshold = rerank_threshold
        self._seen_chunks: set[str] = set()
        self._seen_chunks_lock = Lock()

    def execute(
        self,
        args: Any,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        input_data = self._parse_args(args)

        user_specified_kbs = input_data.knowledge_base_ids
        search_targets = self.search_targets
        if user_specified_kbs:
            allowed = set(user_specified_kbs)
            search_targets = SearchTargets(
                [target for target in self.search_targets if target.knowledge_base_id in allowed]
            )

        if len(search_targets) == 0:
            result = ToolResult(
                success=False,
                error="no knowledge bases specified and no search targets configured",
            )
            raise ToolExecutionError("no search targets available", result)

        queries = input_data.queries
        if len(queries) == 0:
            result = ToolResult(success=False, error="queries parameter is required")
            raise ToolExecutionError("no queries provided", result)

        all_results, branch_failures, branch_count = self._run_searches(
            queries,
            search_targets,
            cancellation_token=cancellation_token,
        )
        deduplicated = self._deduplicate_results(all_results)
        deduplicated.sort(key=lambda item: (-item.result.score, item.result.knowledge_id))
        if self.top_k > 0 and len(deduplicated) > self.top_k:
            deduplicated = deduplicated[: self.top_k]

        if len(deduplicated) == 0 and branch_count > 0 and len(branch_failures) == branch_count:
            return self._format_all_branches_failed(
                failures=branch_failures,
                kbs_to_search=search_targets.get_all_knowledge_base_ids(),
                queries=queries,
            )

        return self._format_output(
            results=deduplicated,
            kbs_to_search=search_targets.get_all_knowledge_base_ids(),
            queries=queries,
            branch_failures=branch_failures,
        )

    def _parse_args(self, args: Any) -> KnowledgeSearchInput:
        if isinstance(args, bytes):
            args = args.decode("utf-8")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError) as exc:
                result = ToolResult(success=False, error=f"Failed to parse args: {exc}")
                raise ToolExecutionError(f"Failed to parse args: {exc}", result) from exc
        if not isinstance(args, dict):
            result = ToolResult(
                success=False,
                error=f"Failed to parse args: expected object, got {type(args).__name__}",
            )
            raise ToolExecutionError(result.error, result)

        queries = self._clean_string_array(
            args.get("queries"),
            name="queries",
            max_items=5,
            required=True,
        )
        knowledge_base_ids = self._clean_string_array(
            args.get("knowledge_base_ids") or [],
            name="knowledge_base_ids",
            max_items=10,
            required=False,
        )
        return KnowledgeSearchInput(
            queries=queries,
            knowledge_base_ids=knowledge_base_ids,
        )

    def _clean_string_array(
        self,
        raw: Any,
        *,
        name: str,
        max_items: int,
        required: bool,
    ) -> list[str]:
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            result = ToolResult(success=False, error=f"{name} parameter must be an array")
            raise ToolExecutionError(result.error, result)
        if len(raw) > max_items:
            result = ToolResult(
                success=False,
                error=f"{name} parameter must contain at most {max_items} items",
            )
            raise ToolExecutionError(result.error, result)

        cleaned: list[str] = []
        for index, item in enumerate(raw):
            if not isinstance(item, str):
                result = ToolResult(
                    success=False,
                    error=f"{name}[{index}] parameter must be a string",
                )
                raise ToolExecutionError(result.error, result)
            value = item.strip()
            if value:
                cleaned.append(value)

        if required and len(cleaned) == 0:
            result = ToolResult(success=False, error="queries parameter is required")
            raise ToolExecutionError(result.error, result)
        return cleaned

    def _run_searches(
        self,
        queries: Sequence[str],
        search_targets: SearchTargets,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[list[_SearchResultWithMeta], list[_SearchBranchFailure], int]:
        all_results: list[_SearchResultWithMeta] = []
        branch_failures: list[_SearchBranchFailure] = []
        branch_count = 0
        full_kb_targets = [
            target for target in search_targets if not target.knowledge_ids and target.knowledge_base_id
        ]
        knowledge_targets = [
            target for target in search_targets if target.knowledge_ids and target.knowledge_base_id
        ]

        grouped_full_kb_targets = self._group_targets_by_embedding_model(full_kb_targets)
        for query in queries:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            for target_group in grouped_full_kb_targets:
                branch_count += 1
                results, failure = self._run_pipeline(
                    query=query,
                    knowledge_base_ids=target_group.get_all_knowledge_base_ids(),
                    allowed_targets=target_group,
                    cancellation_token=cancellation_token,
                )
                all_results.extend(results)
                if failure is not None:
                    branch_failures.append(failure)

            for target in knowledge_targets:
                branch_count += 1
                results, failure = self._run_pipeline(
                    query=query,
                    knowledge_base_ids=[target.knowledge_base_id],
                    knowledge_ids=list(target.knowledge_ids),
                    allowed_targets=SearchTargets([target]),
                    cancellation_token=cancellation_token,
                )
                all_results.extend(results)
                if failure is not None:
                    branch_failures.append(failure)

        return all_results, branch_failures, branch_count

    def _group_targets_by_embedding_model(
        self, targets: Sequence[SearchTarget]
    ) -> list[SearchTargets]:
        if not targets:
            return []
        model_keys = self._resolve_embedding_model_keys(targets)
        grouped: dict[str, SearchTargets] = {}
        for target in targets:
            key = model_keys.get(target.knowledge_base_id, "")
            grouped.setdefault(key, SearchTargets()).append(target)
        return list(grouped.values())

    def _resolve_embedding_model_keys(
        self, targets: Sequence[SearchTarget]
    ) -> dict[str, str]:
        hybrid_search = getattr(self.search_pipeline, "hybrid_search", None)
        if hybrid_search is None:
            return {}

        kb_map = dict(getattr(hybrid_search, "knowledge_bases", {}) or {})
        default_kb = getattr(hybrid_search, "_default_kb", None)
        kb_refs = []
        for target in targets:
            kb = kb_map.get(target.knowledge_base_id)
            if kb is None and callable(default_kb):
                kb = default_kb(target.knowledge_base_id)
            if kb is not None:
                kb_refs.append(kb)

        resolved_by_id: dict[str, str] = {}
        resolve_models = getattr(hybrid_search, "_resolve_embedding_models", None)
        if callable(resolve_models) and kb_refs:
            try:
                resolved_models = resolve_models(kb_refs)
            except Exception:
                resolved_models = {}
            for kb in kb_refs:
                resolved = resolved_models.get(kb.id) if isinstance(resolved_models, dict) else None
                identity = getattr(resolved, "identity_key", "")
                if identity:
                    resolved_by_id[kb.id] = str(identity)

        for kb in kb_refs:
            if kb.id in resolved_by_id:
                continue
            model_identity = getattr(kb, "model_identity", None)
            if callable(model_identity):
                identity = model_identity()
            else:
                identity = getattr(kb, "embedding_model_key", "") or getattr(
                    kb, "embedding_model_id", ""
                )
            if identity:
                resolved_by_id[kb.id] = str(identity)

        return resolved_by_id

    def _run_pipeline(
        self,
        *,
        query: str,
        knowledge_base_ids: Sequence[str],
        allowed_targets: SearchTargets,
        knowledge_ids: Sequence[str] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[list[_SearchResultWithMeta], _SearchBranchFailure | None]:
        if not knowledge_base_ids:
            return [], None
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        try:
            pipeline_result = self.search_pipeline.run(
                query,
                primary_knowledge_base_id=knowledge_base_ids[0],
                knowledge_base_ids=list(knowledge_base_ids),
                knowledge_ids=list(knowledge_ids or []),
                top_k=self.top_k,
                vector_threshold=self.vector_threshold,
                keyword_threshold=self.keyword_threshold,
                rerank_top_k=self.top_k,
                rerank_threshold=self.rerank_threshold,
                faq_priority_enabled=False,
                cancellation_token=cancellation_token,
            )
        except OperationCancelled:
            raise
        except Exception as exc:
            return [], _SearchBranchFailure(
                query=query,
                knowledge_base_ids=tuple(knowledge_base_ids),
                knowledge_ids=tuple(knowledge_ids or ()),
                error=str(exc),
            )

        allowed_kbs = set(allowed_targets.get_all_knowledge_base_ids())
        allowed_knowledge_ids = {
            target.knowledge_base_id: set(target.knowledge_ids)
            for target in allowed_targets
            if target.knowledge_ids
        }

        out: list[_SearchResultWithMeta] = []
        for result in list(getattr(pipeline_result, "results", []) or []):
            if result.knowledge_base_id not in allowed_kbs:
                continue
            scoped_knowledge_ids = allowed_knowledge_ids.get(result.knowledge_base_id)
            if scoped_knowledge_ids is not None and result.knowledge_id not in scoped_knowledge_ids:
                continue
            out.append(
                _SearchResultWithMeta(
                    result=result,
                    source_query=query,
                    query_type="hybrid",
                    knowledge_base_id=result.knowledge_base_id,
                    knowledge_base_type=allowed_targets.get_type_for_kb(result.knowledge_base_id),
                )
            )
        return out, None

    def _deduplicate_results(
        self, results: Sequence[_SearchResultWithMeta]
    ) -> list[_SearchResultWithMeta]:
        seen: set[str] = set()
        content_sig: set[str] = set()
        unique_results: list[_SearchResultWithMeta] = []

        for item in results:
            result = item.result
            keys = [result.id]
            if result.parent_chunk_id:
                keys.append("parent:" + result.parent_chunk_id)
            if result.knowledge_id:
                keys.append(f"kb:{result.knowledge_id}#{result.chunk_index}")

            if any(key in seen for key in keys):
                continue

            sig = build_content_signature(result.content)
            if sig:
                if sig in content_sig:
                    continue
                content_sig.add(sig)

            for key in keys:
                seen.add(key)
            unique_results.append(item)

        seen_by_id: dict[str, _SearchResultWithMeta] = {}
        for item in unique_results:
            existing = seen_by_id.get(item.result.id)
            if existing is None or item.result.score > existing.result.score:
                seen_by_id[item.result.id] = item
        return list(seen_by_id.values())

    def _format_output(
        self,
        *,
        results: Sequence[_SearchResultWithMeta],
        kbs_to_search: Sequence[str],
        queries: Sequence[str],
        branch_failures: Sequence[_SearchBranchFailure] = (),
    ) -> ToolResult:
        if len(results) == 0:
            data: dict[str, Any] = {
                "knowledge_base_ids": list(kbs_to_search),
                "results": [],
                "count": 0,
            }
            if len(queries) > 0:
                data["queries"] = list(queries)
            if branch_failures:
                data["branch_failures"] = [failure.to_dict() for failure in branch_failures]
            output = f"No relevant content found in {len(kbs_to_search)} knowledge base(s).\n\n"
            if branch_failures:
                output += _format_branch_failure_warning(branch_failures) + "\n\n"
            output += "=== \u26a0\ufe0f CRITICAL - Next Steps ===\n"
            output += "- \u274c DO NOT use training data or general knowledge to answer\n"
            output += "- \u2705 State 'I couldn't find relevant information in the knowledge base'\n"
            output += "- NEVER fabricate or infer answers - ONLY use retrieved content\n"
            return ToolResult(success=True, output=output, data=data)

        kb_counts: dict[str, int] = {}
        for item in results:
            kb_counts[item.result.knowledge_id] = kb_counts.get(item.result.knowledge_id, 0) + 1

        out: list[str] = [f'<search_results count="{len(results)}">']
        for query in queries:
            out.append(f"<query>{xml_escape(query)}</query>")

        formatted_results: list[dict[str, Any]] = []
        knowledge_chunk_map: dict[str, set[int]] = {}
        knowledge_title_map: dict[str, str] = {}

        for index, item in enumerate(results, start=1):
            result = item.result
            knowledge_chunk_map.setdefault(result.knowledge_id, set()).add(result.chunk_index)
            knowledge_title_map[result.knowledge_id] = result.knowledge_title

            is_faq = (
                item.knowledge_base_type == KNOWLEDGE_BASE_TYPE_FAQ
                or result.chunk_type == CHUNK_TYPE_FAQ
            )
            with self._seen_chunks_lock:
                seen = result.id in self._seen_chunks
                self._seen_chunks.add(result.id)

            if is_faq:
                attrs = (
                    f'rank="{index}" faq_id="{xml_escape(result.id)}" '
                    f'index="{result.chunk_index}" '
                    f'knowledge_base_id="{xml_escape(result.knowledge_base_id)}" '
                    f'knowledge_title="{xml_escape(result.knowledge_title)}" '
                    f'score="{result.score:.3f}" source_query="{xml_escape(item.source_query)}"'
                )
                tag_name = "faq"
            else:
                attrs = (
                    f'rank="{index}" chunk_id="{xml_escape(result.id)}" '
                    f'chunk_index="{result.chunk_index}" '
                    f'knowledge_id="{xml_escape(result.knowledge_id)}" '
                    f'knowledge_base_id="{xml_escape(result.knowledge_base_id)}" '
                    f'knowledge_title="{xml_escape(result.knowledge_title)}" '
                    f'score="{result.score:.3f}" source_query="{xml_escape(item.source_query)}"'
                )
                tag_name = "chunk"

            if seen:
                out.append(f"<{tag_name} {attrs} already_seen=\"true\">")
                out.append(
                    "<note>(content omitted, already returned in a previous knowledge_search call this session)</note>"
                )
                out.append(f"</{tag_name}>")
            else:
                out.append(f"<{tag_name} {attrs}>")
                snippet = extract_snippet_for_queries(result.content, list(queries))
                if snippet:
                    out.append(f"<match_snippet>{xml_escape(snippet)}</match_snippet>")
                out.append(f"<content>{result.content}</content>")
                for image in _load_image_infos(result.image_info):
                    url = str(image.get("url", ""))
                    if not url:
                        continue
                    out.append(f'<image url="{xml_escape(url)}">')
                    caption = str(image.get("caption", ""))
                    ocr_text = str(image.get("ocr_text", "") or image.get("ocrText", ""))
                    if caption:
                        out.append(f"<image_caption>{xml_escape(caption)}</image_caption>")
                    if ocr_text:
                        out.append(f"<image_ocr>{xml_escape(ocr_text)}</image_ocr>")
                    out.append("</image>")
                out.append(f"</{tag_name}>")

            chunk_data: dict[str, Any] = {
                "result_index": index,
                "content": result.content,
                "knowledge_id": result.knowledge_id,
                "knowledge_title": result.knowledge_title,
                "match_type": result.match_type,
                "source_query": item.source_query,
                "query_type": item.query_type,
                "knowledge_base_type": item.knowledge_base_type,
            }
            images = _format_images_for_data(result.image_info)
            if images:
                chunk_data["images"] = images
            if is_faq:
                chunk_data["faq_id"] = result.id
                chunk_data["index"] = result.chunk_index
            else:
                chunk_data["chunk_id"] = result.id
                chunk_data["chunk_index"] = result.chunk_index
            formatted_results.append(chunk_data)

        out.append("<retrieval_statistics>")
        out.append("</retrieval_statistics>")
        if branch_failures:
            out.append("<retrieval_warnings>")
            for failure in branch_failures:
                kb_ids = ",".join(failure.knowledge_base_ids)
                knowledge_ids = ",".join(failure.knowledge_ids)
                out.append(
                    f'<branch_failure query="{xml_escape(failure.query)}" '
                    f'knowledge_base_ids="{xml_escape(kb_ids)}" '
                    f'knowledge_ids="{xml_escape(knowledge_ids)}">'
                )
                out.append(xml_escape(failure.error))
                out.append("</branch_failure>")
            out.append("</retrieval_warnings>")
        out.append("</search_results>")

        data = {
            "knowledge_base_ids": list(kbs_to_search),
            "results": formatted_results,
            "count": len(formatted_results),
            "kb_counts": kb_counts,
            "display_type": "search_results",
        }
        if len(queries) > 0:
            data["queries"] = list(queries)
        if branch_failures:
            data["branch_failures"] = [failure.to_dict() for failure in branch_failures]

        return ToolResult(success=True, output="\n".join(out), data=data)

    def _format_all_branches_failed(
        self,
        *,
        failures: Sequence[_SearchBranchFailure],
        kbs_to_search: Sequence[str],
        queries: Sequence[str],
    ) -> ToolResult:
        return ToolResult(
            success=False,
            output=_format_branch_failure_warning(failures),
            error="All retrieval branches failed; no knowledge_search results were produced.",
            data={
                "knowledge_base_ids": list(kbs_to_search),
                "queries": list(queries),
                "results": [],
                "count": 0,
                "branch_failures": [failure.to_dict() for failure in failures],
            },
        )

    def _normalize_search_targets(
        self,
        *,
        search_targets: Sequence[SearchTarget | str | dict[str, Any]] | None,
        allowed_knowledge_base_ids: Sequence[str] | None,
        knowledge_base_types: dict[str, str],
    ) -> SearchTargets:
        raw_targets = list(search_targets or [])
        if not raw_targets and allowed_knowledge_base_ids:
            raw_targets = list(allowed_knowledge_base_ids)
        if not raw_targets:
            raw_targets = self._targets_from_pipeline()

        targets = SearchTargets()
        for raw in raw_targets:
            target = self._coerce_search_target(raw, knowledge_base_types)
            if target is not None and target.knowledge_base_id:
                targets.append(target)
        return targets

    def _targets_from_pipeline(self) -> list[str]:
        hybrid_search = getattr(self.search_pipeline, "hybrid_search", None)
        knowledge_bases = getattr(hybrid_search, "knowledge_bases", None)
        if isinstance(knowledge_bases, dict):
            return list(knowledge_bases.keys())
        return []

    def _coerce_search_target(
        self,
        raw: SearchTarget | str | dict[str, Any],
        knowledge_base_types: dict[str, str],
    ) -> SearchTarget | None:
        if isinstance(raw, SearchTarget):
            if raw.knowledge_base_type:
                return raw
            return SearchTarget(
                knowledge_base_id=raw.knowledge_base_id,
                knowledge_ids=tuple(raw.knowledge_ids),
                tenant_id=raw.tenant_id,
                target_type=raw.target_type,
                knowledge_base_type=knowledge_base_types.get(raw.knowledge_base_id, ""),
            )
        if isinstance(raw, str):
            return SearchTarget(
                knowledge_base_id=raw,
                knowledge_base_type=knowledge_base_types.get(raw, ""),
            )
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
                    raw.get("knowledge_base_type")
                    or raw.get("KnowledgeBaseType")
                    or knowledge_base_types.get(kb_id, "")
                ),
            )
        return None


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _format_branch_failure_warning(failures: Sequence[_SearchBranchFailure]) -> str:
    lines = ["Retrieval branch failures:"]
    for failure in failures:
        kb_ids = ", ".join(failure.knowledge_base_ids) or "(none)"
        knowledge_ids = ", ".join(failure.knowledge_ids)
        scope = f"KBs [{kb_ids}]"
        if knowledge_ids:
            scope += f", knowledge_ids [{knowledge_ids}]"
        lines.append(f"- query={failure.query!r}, {scope}: {failure.error}")
    return "\n".join(lines)


snippetContextRunes = 200


def extract_snippet_for_queries(content: str, queries: Sequence[str]) -> str:
    content = content.strip()
    if content == "":
        return ""

    tokens = search_query_tokens(queries)
    lowered = content.lower()
    earliest = -1
    earliest_end = -1
    for token in tokens:
        index = lowered.find(token)
        if index < 0:
            continue
        end = index + len(token)
        if earliest < 0 or index < earliest:
            earliest = index
            earliest_end = end

    if earliest < 0:
        if len(content) > snippetContextRunes * 2:
            return content[: snippetContextRunes * 2].strip() + " ..."
        return content

    match_text = content[earliest:earliest_end]
    before = content[:earliest]
    after = content[earliest_end:]
    if len(before) > snippetContextRunes:
        before = before[-snippetContextRunes:]
    if len(after) > snippetContextRunes:
        after = after[:snippetContextRunes]

    snippet = before + match_text + after
    snippet = snippet.replace("\n", " ")
    while "  " in snippet:
        snippet = snippet.replace("  ", " ")
    return "... " + snippet.strip() + " ..."


def search_query_tokens(queries: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    separators = set(" \t\n\r,.;:?!()[]{}\"'")
    for query in queries:
        current: list[str] = []
        for char in query:
            if char in separators:
                token = "".join(current).strip().lower()
                if len(token) >= 2 and token not in seen:
                    seen.add(token)
                    tokens.append(token)
                current = []
            else:
                current.append(char)
        token = "".join(current).strip().lower()
        if len(token) >= 2 and token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _load_image_infos(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _format_images_for_data(raw: str) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for image in _load_image_infos(raw):
        image_data: dict[str, str] = {}
        if image.get("url"):
            image_data["url"] = str(image["url"])
        if image.get("caption"):
            image_data["caption"] = str(image["caption"])
        ocr_text = image.get("ocr_text") or image.get("ocrText")
        if ocr_text:
            image_data["ocr_text"] = str(ocr_text)
        if image_data:
            images.append(image_data)
    return images
