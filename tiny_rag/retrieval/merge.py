from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from tiny_rag.cancellation import CancellationToken, call_with_cancellation
from tiny_rag.persistence.chunk_models import ChunkRow

from .models import (
    CHUNK_TYPE_FAQ,
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_PARENT_TEXT,
    CHUNK_TYPE_TEXT,
    MATCH_TYPE_HISTORY,
    SearchResult,
)
from .utils import (
    append_with_overlap,
    build_content_signature,
    concat_no_overlap,
    content_overlap_ratio,
    filter_image_info_by_content_urls,
    filter_image_info_by_match_range,
    is_content_contained,
    jaccard,
    merge_image_info_json,
    merge_ordered_content,
    normalize_content,
    prune_markdown_images_outside_range,
    slice_content_by_document_range,
    tokenize_simple,
)


class MergeService:
    def __init__(self, chunk_repository: object | None = None, *, tenant_id: int = 0) -> None:
        self.chunk_repository = chunk_repository
        self.tenant_id = tenant_id

    def merge(
        self,
        results: Sequence[SearchResult] | None = None,
        *,
        search_results: Sequence[SearchResult] | None = None,
        rerank_results: Sequence[SearchResult] | None = None,
        history_references: Sequence[SearchResult] | None = None,
        query: str = "",
        rewrite_query: str = "",
        tenant_id: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[SearchResult]:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        previous_token = getattr(self, "_active_cancellation_token", None)
        self._active_cancellation_token = cancellation_token
        try:
            return self._merge_with_cancellation(
                results=results,
                search_results=search_results,
                rerank_results=rerank_results,
                history_references=history_references,
                query=query,
                rewrite_query=rewrite_query,
                tenant_id=tenant_id,
                cancellation_token=cancellation_token,
            )
        finally:
            self._active_cancellation_token = previous_token

    def _merge_with_cancellation(
        self,
        results: Sequence[SearchResult] | None = None,
        *,
        search_results: Sequence[SearchResult] | None = None,
        rerank_results: Sequence[SearchResult] | None = None,
        history_references: Sequence[SearchResult] | None = None,
        query: str = "",
        rewrite_query: str = "",
        tenant_id: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[SearchResult]:
        effective_tenant_id = self.tenant_id if tenant_id is None else tenant_id
        selected = self._select_input_results(results, search_results, rerank_results)
        selected = remove_duplicate_results(selected)
        selected = self._inject_history_results(
            current=selected,
            history_references=list(history_references or []),
            query=query,
            rewrite_query=rewrite_query,
        )
        if not selected:
            return []

        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        selected = self.resolve_parent_chunks(selected, effective_tenant_id)
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        merged = self.group_and_merge_overlapping(selected)
        merged = self.populate_faq_answers(merged, effective_tenant_id)
        merged = self.expand_short_context_with_neighbors(merged, effective_tenant_id)
        merged = self.group_and_merge_overlapping(merged)
        merged = remove_duplicate_results(merged)
        return remove_partial_overlaps(merged)

    def _select_input_results(
        self,
        results: Sequence[SearchResult] | None,
        search_results: Sequence[SearchResult] | None,
        rerank_results: Sequence[SearchResult] | None,
    ) -> list[SearchResult]:
        if rerank_results:
            return list(rerank_results)
        if results is not None:
            return list(results)
        fallback = list(search_results or [])
        fallback.sort(key=lambda item: item.score, reverse=True)
        return fallback

    def _inject_history_results(
        self,
        *,
        current: list[SearchResult],
        history_references: list[SearchResult],
        query: str,
        rewrite_query: str,
    ) -> list[SearchResult]:
        history = filter_history_results(
            query=rewrite_query or query,
            history_references=history_references,
            current_results=current,
        )
        if not history:
            return current
        return remove_duplicate_results(current + history)

    def resolve_parent_chunks(
        self,
        results: Sequence[SearchResult],
        tenant_id: int | None = None,
    ) -> list[SearchResult]:
        if not results or self.chunk_repository is None:
            return list(results)
        effective_tenant_id = self.tenant_id if tenant_id is None else tenant_id
        if effective_tenant_id == 0:
            return list(results)

        parent_ids = sorted({result.parent_chunk_id for result in results if result.parent_chunk_id})
        if not parent_ids:
            return list(results)
        parent_chunks = self._list_chunks_by_id(effective_tenant_id, parent_ids)
        parent_map = {chunk.id: chunk for chunk in parent_chunks}

        has_image_results = any(
            result.chunk_type in {CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION}
            for result in results
        )
        if has_image_results:
            grandparent_ids = [
                chunk.parent_chunk_id
                for chunk in parent_chunks
                if chunk.parent_chunk_id
                and chunk.chunk_type == CHUNK_TYPE_TEXT
                and chunk.parent_chunk_id not in parent_map
            ]
            if grandparent_ids:
                for chunk in self._list_chunks_by_id(effective_tenant_id, grandparent_ids):
                    parent_map[chunk.id] = chunk

        text_child_ids = collect_scoped_text_child_ids(results, parent_map)
        scoped_image_info = (
            self.collect_image_info_by_chunk_ids(effective_tenant_id, text_child_ids)
            if text_child_ids
            else {}
        )

        output = list(results)
        for result in output:
            if result.parent_chunk_id == "":
                continue

            if result.chunk_type == CHUNK_TYPE_TEXT:
                parent = parent_map.get(result.parent_chunk_id)
                if (
                    parent is None
                    or parent.content == ""
                    or parent.chunk_type != CHUNK_TYPE_PARENT_TEXT
                ):
                    continue
                match_start, match_end = result.start_at, result.end_at
                result.content = prune_markdown_images_outside_range(
                    parent.content,
                    parent.start_at,
                    match_start,
                    match_end,
                )
                result.start_at = parent.start_at
                result.end_at = parent.end_at
                assign_scoped_image_info(result, scoped_image_info, result.id)
                if result.image_info:
                    result.image_info = filter_image_info_by_match_range(
                        parent.content,
                        parent.start_at,
                        match_start,
                        match_end,
                        result.image_info,
                    )
                if result.id not in result.sub_chunk_id:
                    result.sub_chunk_id.append(result.id)
                continue

            if result.chunk_type in {CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION}:
                text_parent = parent_map.get(result.parent_chunk_id)
                if (
                    text_parent is None
                    or text_parent.content == ""
                    or text_parent.chunk_type != CHUNK_TYPE_TEXT
                ):
                    continue
                hit_image_info = result.image_info
                content_source = text_parent
                if text_parent.parent_chunk_id:
                    grandparent = parent_map.get(text_parent.parent_chunk_id)
                    if (
                        grandparent is not None
                        and grandparent.chunk_type == CHUNK_TYPE_PARENT_TEXT
                        and grandparent.content
                    ):
                        content_source = grandparent
                match_start, match_end = text_parent.start_at, text_parent.end_at
                sliced = slice_content_by_document_range(
                    content_source.content,
                    content_source.start_at,
                    match_start,
                    match_end,
                )
                if sliced == "":
                    sliced = text_parent.content
                    match_start, match_end = text_parent.start_at, text_parent.end_at
                result.content = sliced
                result.start_at = match_start
                result.end_at = match_end
                assign_scoped_image_info(result, scoped_image_info, text_parent.id)
                if result.image_info == "" and hit_image_info:
                    result.image_info = filter_image_info_by_content_urls(result.content, hit_image_info)
                if result.id not in result.sub_chunk_id:
                    result.sub_chunk_id.append(result.id)

        return output

    def group_and_merge_overlapping(
        self, results: Sequence[SearchResult]
    ) -> list[SearchResult]:
        knowledge_group: dict[str, dict[str, list[SearchResult]]] = {}
        for chunk in results:
            knowledge_group.setdefault(chunk.knowledge_id, {}).setdefault(
                chunk.chunk_type, []
            ).append(chunk)

        merged_chunks: list[SearchResult] = []
        for chunk_group in knowledge_group.values():
            for chunks in chunk_group.values():
                chunks.sort(key=lambda item: (item.start_at, item.end_at))
                merged_chunks.extend(self.merge_overlapping_chunks(chunks))
        return merged_chunks

    def merge_overlapping_chunks(
        self, chunks: Sequence[SearchResult]
    ) -> list[SearchResult]:
        if not chunks:
            return []
        merged = [chunks[0]]
        for current in chunks[1:]:
            last = merged[-1]
            if current.start_at > last.end_at:
                merged.append(current)
                continue

            if current.end_at > last.end_at:
                last.content = append_with_overlap(
                    last.content,
                    current.content,
                    last.end_at - current.start_at,
                )
                last.end_at = current.end_at
                last.sub_chunk_id.append(current.id)
                last.image_info = merge_image_info_json(last.image_info, current.image_info)
            else:
                if current.id not in last.sub_chunk_id:
                    last.sub_chunk_id.append(current.id)
                last.image_info = merge_image_info_json(last.image_info, current.image_info)

            if current.score > last.score:
                last.score = current.score

        merged.sort(key=lambda item: item.score, reverse=True)
        return merged

    def populate_faq_answers(
        self,
        results: Sequence[SearchResult],
        tenant_id: int | None = None,
    ) -> list[SearchResult]:
        if not results or self.chunk_repository is None:
            return list(results)
        effective_tenant_id = self.tenant_id if tenant_id is None else tenant_id
        if effective_tenant_id == 0:
            return list(results)

        chunk_result_map: dict[str, list[SearchResult]] = {}
        for result in results:
            if result.id and result.chunk_type == CHUNK_TYPE_FAQ:
                chunk_result_map.setdefault(result.id, []).append(result)
        if not chunk_result_map:
            return list(results)

        chunks = self._list_chunks_by_id(effective_tenant_id, list(chunk_result_map))
        for chunk in chunks:
            meta = parse_faq_metadata(chunk.metadata)
            content = build_faq_answer_content(meta)
            if content == "":
                continue
            for result in chunk_result_map.get(chunk.id, []):
                result.content = content
        return list(results)

    def expand_short_context_with_neighbors(
        self,
        results: Sequence[SearchResult],
        tenant_id: int | None = None,
    ) -> list[SearchResult]:
        min_len = 350
        max_len = 850
        if not results or self.chunk_repository is None:
            return list(results)
        effective_tenant_id = self.tenant_id if tenant_id is None else tenant_id
        if effective_tenant_id == 0:
            return list(results)

        targets = [
            result
            for result in results
            if result is not None
            and result.id
            and result.content
            and result.chunk_type == CHUNK_TYPE_TEXT
            and len(result.content) < min_len
        ]
        if not targets:
            return list(results)

        base_ids = list(dict.fromkeys(result.id for result in targets))
        chunk_map = {chunk.id: chunk for chunk in self._list_chunks_by_id(effective_tenant_id, base_ids)}

        neighbor_ids: set[str] = set()
        for chunk in chunk_map.values():
            if chunk.pre_chunk_id and chunk.pre_chunk_id not in chunk_map:
                neighbor_ids.add(chunk.pre_chunk_id)
            if chunk.next_chunk_id and chunk.next_chunk_id not in chunk_map:
                neighbor_ids.add(chunk.next_chunk_id)
        if neighbor_ids:
            for chunk in self._list_chunks_by_id(effective_tenant_id, sorted(neighbor_ids)):
                chunk_map[chunk.id] = chunk

        for result in targets:
            self.fetch_chunks_if_missing(effective_tenant_id, chunk_map, result.id)
            base_chunk = chunk_map.get(result.id)
            if (
                base_chunk is None
                or base_chunk.content == ""
                or base_chunk.chunk_type != CHUNK_TYPE_TEXT
            ):
                continue

            prev_content = ""
            next_content = ""
            prev_ids: list[str] = []
            next_ids: list[str] = []

            prev_cursor = base_chunk.pre_chunk_id
            next_cursor = base_chunk.next_chunk_id
            self.fetch_chunks_if_missing(effective_tenant_id, chunk_map, prev_cursor, next_cursor)

            if prev_cursor:
                prev_chunk = chunk_map.get(prev_cursor)
                if prev_chunk is not None and prev_chunk.knowledge_id == base_chunk.knowledge_id:
                    prev_content = prev_chunk.content
                    prev_ids.append(prev_chunk.id)
                    prev_cursor = prev_chunk.pre_chunk_id
                else:
                    prev_cursor = ""

            if next_cursor:
                next_chunk = chunk_map.get(next_cursor)
                if next_chunk is not None and next_chunk.knowledge_id == base_chunk.knowledge_id:
                    next_content = next_chunk.content
                    next_ids.append(next_chunk.id)
                    next_cursor = next_chunk.next_chunk_id
                else:
                    next_cursor = ""

            while True:
                merged = merge_ordered_content(prev_content, base_chunk.content, next_content, max_len)
                if merged == "" or len(merged) >= min_len or (prev_cursor == "" and next_cursor == ""):
                    break

                expanded = False
                if prev_cursor:
                    self.fetch_chunks_if_missing(effective_tenant_id, chunk_map, prev_cursor)
                    prev_chunk = chunk_map.get(prev_cursor)
                    if prev_chunk is not None and prev_chunk.knowledge_id == base_chunk.knowledge_id:
                        prev_content = concat_no_overlap(prev_chunk.content, prev_content)
                        prev_ids.insert(0, prev_chunk.id)
                        prev_cursor = prev_chunk.pre_chunk_id
                        expanded = True
                    else:
                        prev_cursor = ""

                merged = merge_ordered_content(prev_content, base_chunk.content, next_content, max_len)
                if len(merged) >= min_len:
                    break

                if next_cursor:
                    self.fetch_chunks_if_missing(effective_tenant_id, chunk_map, next_cursor)
                    next_chunk = chunk_map.get(next_cursor)
                    if next_chunk is not None and next_chunk.knowledge_id == base_chunk.knowledge_id:
                        next_content = concat_no_overlap(next_content, next_chunk.content)
                        next_ids.append(next_chunk.id)
                        next_cursor = next_chunk.next_chunk_id
                        expanded = True
                    else:
                        next_cursor = ""

                if not expanded:
                    break

            merged = merge_ordered_content(prev_content, base_chunk.content, next_content, max_len)
            if merged == "":
                continue
            result.content = merged
            for chunk_id in prev_ids + next_ids:
                if chunk_id and chunk_id not in result.sub_chunk_id:
                    result.sub_chunk_id.append(chunk_id)
            if prev_content:
                result.start_at = base_chunk.start_at - len(prev_content)
                if result.start_at < 0:
                    result.start_at = 0
            result.end_at = result.start_at + len(result.content)

        return list(results)

    def fetch_chunks_if_missing(
        self,
        tenant_id: int,
        chunk_map: dict[str, ChunkRow | None],
        *chunk_ids: str,
    ) -> None:
        missing = [
            chunk_id for chunk_id in chunk_ids if chunk_id and chunk_id not in chunk_map
        ]
        if not missing:
            return
        chunks = self._list_chunks_by_id(tenant_id, missing)
        found = {chunk.id for chunk in chunks}
        for chunk in chunks:
            chunk_map[chunk.id] = chunk
        for chunk_id in missing:
            if chunk_id not in found:
                chunk_map[chunk_id] = None

    def collect_image_info_by_chunk_ids(self, tenant_id: int, chunk_ids: Sequence[str]) -> dict[str, str]:
        if self.chunk_repository is None or not chunk_ids:
            return {}
        children = self._list_chunks_by_parent_ids(tenant_id, chunk_ids)
        if not children:
            return {}

        agg_map: dict[str, dict[str, dict[str, Any]]] = {}

        def add_info(target_id: str, child: ChunkRow) -> None:
            if child.image_info == "":
                return
            infos = _load_image_infos(child.image_info)
            if not infos:
                return
            bucket = agg_map.setdefault(target_id, {})
            for info in infos:
                key = str(info.get("url", "") or info.get("original_url", ""))
                if key == "":
                    continue
                existing = bucket.get(key)
                if existing is None:
                    bucket[key] = dict(info)
                else:
                    if info.get("ocr_text"):
                        existing["ocr_text"] = info["ocr_text"]
                    if info.get("caption"):
                        existing["caption"] = info["caption"]

        text_child_ids: list[str] = []
        text_to_parent: dict[str, str] = {}
        for child in children:
            if child.chunk_type in {CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION}:
                add_info(child.parent_chunk_id, child)
            elif child.chunk_type == CHUNK_TYPE_TEXT:
                text_child_ids.append(child.id)
                text_to_parent[child.id] = child.parent_chunk_id

        if text_child_ids:
            grandchildren = self._list_chunks_by_parent_ids(tenant_id, text_child_ids)
            for grandchild in grandchildren:
                if grandchild.chunk_type not in {CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION}:
                    continue
                parent_text_id = text_to_parent.get(grandchild.parent_chunk_id)
                if parent_text_id:
                    add_info(parent_text_id, grandchild)

        out: dict[str, str] = {}
        for chunk_id, by_url in agg_map.items():
            if by_url:
                out[chunk_id] = json.dumps(
                    list(by_url.values()), ensure_ascii=False, separators=(",", ":")
                )
        return out

    def _list_chunks_by_id(self, tenant_id: int, chunk_ids: Sequence[str]) -> list[ChunkRow]:
        if self.chunk_repository is None or not chunk_ids:
            return []
        cancellation_token = getattr(self, "_active_cancellation_token", None)
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        return list(
            call_with_cancellation(
                self.chunk_repository.list_chunks_by_id,
                tenant_id=tenant_id,
                chunk_ids=list(dict.fromkeys(chunk_ids)),
                cancellation_token=cancellation_token,
            )
        )

    def _list_chunks_by_parent_ids(self, tenant_id: int, parent_ids: Sequence[str]) -> list[ChunkRow]:
        if self.chunk_repository is None or not parent_ids:
            return []
        if not hasattr(self.chunk_repository, "list_chunks_by_parent_ids"):
            return []
        cancellation_token = getattr(self, "_active_cancellation_token", None)
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        return list(
            call_with_cancellation(
                self.chunk_repository.list_chunks_by_parent_ids,
                tenant_id=tenant_id,
                parent_ids=list(dict.fromkeys(parent_ids)),
                cancellation_token=cancellation_token,
            )
        )


def remove_duplicate_results(results: Sequence[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    content_sig: dict[str, str] = {}
    unique: list[SearchResult] = []
    for result in results:
        if result.id in seen:
            continue
        sig = build_content_signature(result.content)
        if sig:
            if sig in content_sig:
                continue
            content_sig[sig] = result.id
        seen.add(result.id)
        unique.append(result)
    return unique


def filter_history_results(
    *,
    query: str,
    history_references: Sequence[SearchResult],
    current_results: Sequence[SearchResult],
) -> list[SearchResult]:
    min_similarity = 0.15
    history_score_discount = 0.6
    max_history_results = 3
    if not history_references:
        return []
    existing_ids = {result.id for result in current_results}
    query_tokens = tokenize_simple(query)

    filtered: list[SearchResult] = []
    for result in history_references:
        if result.id in existing_ids:
            continue
        sim = jaccard(query_tokens, tokenize_simple(result.content))
        if sim < min_similarity:
            continue
        result.match_type = MATCH_TYPE_HISTORY
        result.score *= history_score_discount
        result.metadata = result.metadata or {}
        result.metadata["history_similarity"] = f"{sim:.4f}".rstrip("0").rstrip(".")
        filtered.append(result)
        if len(filtered) >= max_history_results:
            break
    return filtered


def collect_scoped_text_child_ids(
    results: Sequence[SearchResult],
    parent_map: dict[str, ChunkRow],
) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for result in results:
        if result.parent_chunk_id == "":
            continue
        if result.chunk_type == CHUNK_TYPE_TEXT:
            parent = parent_map.get(result.parent_chunk_id)
            if parent is None or parent.chunk_type != CHUNK_TYPE_PARENT_TEXT:
                continue
            if result.id in seen:
                continue
            seen.add(result.id)
            ids.append(result.id)
        elif result.chunk_type in {CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION}:
            if result.parent_chunk_id in seen:
                continue
            seen.add(result.parent_chunk_id)
            ids.append(result.parent_chunk_id)
    return ids


def assign_scoped_image_info(
    result: SearchResult, scoped: dict[str, str] | None, text_child_id: str
) -> None:
    if scoped:
        info = scoped.get(text_child_id, "")
        if info:
            result.image_info = info
            return
    if result.image_info:
        result.image_info = filter_image_info_by_content_urls(result.content, result.image_info)


def parse_faq_metadata(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def build_faq_answer_content(meta: dict[str, Any]) -> str:
    if not meta:
        return ""
    question = str(meta.get("standard_question", meta.get("StandardQuestion", ""))).strip()
    answers_raw = meta.get("answers", meta.get("Answers", []))
    answers: list[str] = []
    if isinstance(answers_raw, str):
        answers_raw = [answers_raw]
    if isinstance(answers_raw, Iterable):
        for answer in answers_raw:
            trimmed = str(answer).strip()
            if trimmed:
                answers.append(trimmed)
    if question == "" and not answers:
        return ""

    lines: list[str] = []
    if question:
        lines.append(f"Q: {question}")
    if answers:
        lines.append("Answer:")
        for answer in answers:
            lines.append(f"- {answer}")
    return "\n".join(lines).strip()


def remove_partial_overlaps(results: Sequence[SearchResult]) -> list[SearchResult]:
    overlap_threshold = 0.85
    if len(results) <= 1:
        return list(results)

    entries = [(normalize_content(result.content), result) for result in results]
    removed: set[int] = set()

    for i in range(len(entries)):
        if i in removed:
            continue
        for j in range(i + 1, len(entries)):
            if j in removed:
                continue
            a_norm, _ = entries[i]
            b_norm, _ = entries[j]
            short_idx, long_idx = (i, j)
            if len(a_norm) > len(b_norm):
                short_idx, long_idx = j, i

            contained = is_content_contained(entries[short_idx][0], entries[long_idx][0])
            if not contained:
                ratio = content_overlap_ratio(
                    entries[short_idx][1].content,
                    entries[long_idx][1].content,
                )
                if ratio < overlap_threshold:
                    continue

            victim = short_idx
            if entries[short_idx][1].score > entries[long_idx][1].score:
                victim = long_idx
            removed.add(victim)

    return [result for index, (_, result) in enumerate(entries) if index not in removed]


def _load_image_infos(raw: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
