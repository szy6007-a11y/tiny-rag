from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable, Sequence

from tiny_rag.cancellation import CancellationToken
from tiny_rag.persistence.chunk_models import ChunkRow
from tiny_rag.persistence.chunk_repository import INSERT_COLUMNS
from tiny_rag.retrieval.merge import parse_faq_metadata
from tiny_rag.retrieval.models import CHUNK_TYPE_FAQ
from tiny_rag.retrieval.utils import build_content_signature, jaccard, tokenize_simple

from .definitions import ToolGrepChunks
from .knowledge_search import SearchTarget, SearchTargets, xml_escape
from .tool import BaseTool, ToolExecutionError, ToolResult


grepChunksDescription = r"""Search knowledge base chunk content with a single POSIX regular expression, applied directly in the database (PostgreSQL ~* / MySQL/SQLite REGEXP, case-insensitive). Behaves like `grep -E -i`.
Pack multiple concepts into ONE regex using `|` alternation — do not call this tool repeatedly for synonyms.
Returns matching chunks with hit counts and a <match_snippet> around the first match (each tagged with its knowledge_id and chunk_id).
Examples:
- Alternation (RECOMMENDED): "stardust|skyvault|psionic" (matches any of the words)
- Multiple terms in order: "psionic.*engine" (matches both words in order)
- Word boundary / anchor: "\\brag\\b" or "^chapter\\s+\\d+"
- Plain text: "engine" (matches literal substring anywhere in chunk content)
IMPORTANT — JSON escaping: every backslash in a regex MUST be written as \\ inside the JSON tool arguments (e.g. to search for literal "C++" write "C\\+\\+", NOT "C\+\+"; for "\d+" write "\\d+"). Plain "\+" / "\d" etc. are invalid JSON escapes and will fail to parse.
Use this to locate candidate chunks by exact identifiers, error codes, product names, or recurring terms.

## Deep read after grep:
- **FAQ hit** (chunk type faq): call list_knowledge_chunks with **faq_id** from the grep result (NOT the parent knowledge_id).
- **Document hit**: call list_knowledge_chunks with **knowledge_id**, or get_document_info with **knowledge_ids**."""


grepChunksSchema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                'A single POSIX regex applied directly to chunk content '
                '(case-insensitive). Combine multiple concepts with "|" alternation '
                'in ONE regex (e.g. "stardust|skyvault|psionic") — do not split '
                "into multiple calls."
            ),
            "minLength": 1,
        }
    },
    "required": ["query"],
}


grepResultLimit = 30
grepMaxFetchLimit = 500
maxKnowledgeRows = 20
snippetContextRunes = 200
snippetMaxMatchRunes = 200
snippetMaxTotalRunes = 800
snippetMaxAnswerRunes = 600


@dataclass(frozen=True)
class GrepChunksInput:
    query: str


@dataclass
class _ChunkWithTitle:
    chunk: ChunkRow
    knowledge_title: str = ""
    match_score: float = 0.0
    matched_patterns: int = 0
    title_match: bool = False
    total_chunk_count: int = 0


class GrepChunksTool(BaseTool):
    def __init__(
        self,
        chunk_repository: object,
        *,
        search_targets: Sequence[SearchTarget | str | dict[str, Any]] | None = None,
        knowledge_titles: dict[str, str] | None = None,
    ) -> None:
        super().__init__(ToolGrepChunks, grepChunksDescription, grepChunksSchema)
        self.chunk_repository = chunk_repository
        self.search_targets = normalize_search_targets(search_targets)
        self.knowledge_titles = dict(knowledge_titles or {})
        self._seen_chunks: dict[str, bool] = {}
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
        query = input_data.query

        try:
            compiled = [re.compile(query, re.IGNORECASE | re.MULTILINE)]
        except re.error as exc:
            result = ToolResult(
                success=False,
                error=f'invalid regex query "{query}": {exc}',
            )
            raise ToolExecutionError(result.error, result) from exc

        queries = [query]
        kb_ids = self.search_targets.get_all_knowledge_base_ids()
        try:
            results = self._search_chunks(
                queries=queries,
                kb_ids=kb_ids,
                targets=self.search_targets,
            )
        except Exception as exc:
            result = ToolResult(success=False, error=f"Search failed: {exc}")
            raise ToolExecutionError(result.error, result) from exc

        deduplicated = self._deduplicate_chunks(results)
        scored = self._score_chunks(deduplicated, compiled)

        final_results = scored
        if len(scored) > 10:
            mmr_k = len(scored)
            if grepResultLimit > 0 and mmr_k > grepResultLimit:
                mmr_k = grepResultLimit
            mmr_results = self._apply_mmr(scored, mmr_k, 0.7)
            if mmr_results:
                final_results = mmr_results

        final_results.sort(
            key=lambda item: (
                not item.title_match,
                -item.matched_patterns,
                -item.match_score,
                item.chunk.chunk_index,
            )
        )
        if len(final_results) > grepResultLimit:
            final_results = final_results[:grepResultLimit]

        chunk_results = build_grep_chunk_results(final_results, compiled)
        aggregated_results = self._aggregate_by_knowledge(final_results, queries, compiled)
        document_count = len(aggregated_results or [])
        knowledge_results_for_ui = aggregated_results
        if knowledge_results_for_ui is not None and len(knowledge_results_for_ui) > maxKnowledgeRows:
            knowledge_results_for_ui = knowledge_results_for_ui[:maxKnowledgeRows]

        return ToolResult(
            success=True,
            output=self._format_output(final_results, queries, compiled),
            data={
                "query": query,
                "queries": queries,
                "patterns": queries,
                "chunk_results": chunk_results,
                "knowledge_results": knowledge_results_for_ui,
                "result_count": len(chunk_results or []),
                "document_count": document_count,
                "total_matches": len(final_results),
                "knowledge_base_ids": kb_ids,
                "limit": grepResultLimit,
                "max_results": grepResultLimit,
                "display_type": "grep_results",
            },
        )

    def _parse_args(self, args: Any) -> GrepChunksInput:
        if isinstance(args, bytes):
            args = args.decode("utf-8")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError) as exc:
                result = ToolResult(success=False, error=f"Failed to parse args: {exc}")
                raise ToolExecutionError(result.error, result) from exc
        if not isinstance(args, dict):
            result = ToolResult(
                success=False,
                error=f"Failed to parse args: expected object, got {type(args).__name__}",
            )
            raise ToolExecutionError(result.error, result)

        query = str(args.get("query") or "").strip()
        if query == "":
            result = ToolResult(
                success=False,
                error="query parameter is required and must be a non-empty regex string",
            )
            raise ToolExecutionError(result.error, result)
        return GrepChunksInput(query=query)

    def _search_chunks(
        self,
        *,
        queries: Sequence[str],
        kb_ids: Sequence[str],
        targets: Sequence[SearchTarget],
    ) -> list[_ChunkWithTitle]:
        conn = getattr(self.chunk_repository, "conn", None)
        if not isinstance(conn, sqlite3.Connection):
            raise TypeError("grep_chunks requires a SQLite-backed ChunkRepository")
        if not kb_ids:
            return []

        _register_regexp(conn)
        _sync_temp_knowledge_titles(conn, self.knowledge_titles)

        select_columns = ", ".join(f"chunks.{column}" for column in INSERT_COLUMNS)
        sql = [
            f"""
            SELECT {select_columns},
                   COALESCE(_tiny_rag_knowledge_titles.title, '') AS knowledge_title
            FROM chunks
            LEFT JOIN _tiny_rag_knowledge_titles
              ON chunks.knowledge_id = _tiny_rag_knowledge_titles.knowledge_id
            WHERE chunks.is_enabled = 1
              AND chunks.deleted_at IS NULL
            """
        ]
        params: list[Any] = []

        scope_conditions: list[str] = []
        for target in targets:
            condition, condition_params = build_chunk_scope_condition(target)
            if condition:
                scope_conditions.append(condition)
                params.extend(condition_params)
        if not scope_conditions:
            return []
        sql.append("AND (" + " OR ".join(scope_conditions) + ")")

        regex_conditions: list[str] = []
        regex_params: list[Any] = []
        for query in queries:
            regex_conditions.append(
                "(chunks.content REGEXP ? OR COALESCE(_tiny_rag_knowledge_titles.title, '') REGEXP ?)"
            )
            regex_params.extend([query, query])
        if regex_conditions:
            sql.append("AND (" + " OR ".join(regex_conditions) + ")")
            params.extend(regex_params)

        sql.append("ORDER BY chunks.created_at DESC LIMIT ?")
        params.append(grepMaxFetchLimit)

        rows = conn.execute("\n".join(sql), tuple(params)).fetchall()
        results: list[_ChunkWithTitle] = []
        for row in rows:
            chunk = self.chunk_repository._from_row(row)
            results.append(
                _ChunkWithTitle(
                    chunk=chunk,
                    knowledge_title=str(row["knowledge_title"] or ""),
                )
            )

        self._populate_total_chunk_counts(conn, results)
        return results

    def _populate_total_chunk_counts(
        self,
        conn: sqlite3.Connection,
        results: Sequence[_ChunkWithTitle],
    ) -> None:
        scope_keys = sorted(
            {
                (item.chunk.tenant_id, item.chunk.knowledge_base_id, item.chunk.knowledge_id)
                for item in results
                if item.chunk.knowledge_id
            }
        )
        if not scope_keys:
            return
        conditions = []
        params: list[Any] = []
        for tenant_id, knowledge_base_id, knowledge_id in scope_keys:
            conditions.append("(tenant_id = ? AND knowledge_base_id = ? AND knowledge_id = ?)")
            params.extend([tenant_id, knowledge_base_id, knowledge_id])
        rows = conn.execute(
            f"""
            SELECT tenant_id, knowledge_base_id, knowledge_id, COUNT(*) AS cnt
            FROM chunks
            WHERE ({" OR ".join(conditions)})
              AND is_enabled = 1
              AND deleted_at IS NULL
            GROUP BY tenant_id, knowledge_base_id, knowledge_id
            """,
            tuple(params),
        ).fetchall()
        count_map = {
            (int(row["tenant_id"]), str(row["knowledge_base_id"]), str(row["knowledge_id"])): int(
                row["cnt"]
            )
            for row in rows
        }
        for item in results:
            item.total_chunk_count = count_map.get(
                (
                    item.chunk.tenant_id,
                    item.chunk.knowledge_base_id,
                    item.chunk.knowledge_id,
                ),
                0,
            )

    def _format_output(
        self,
        results: Sequence[_ChunkWithTitle],
        queries: Sequence[str],
        compiled: Sequence[re.Pattern[str]],
    ) -> str:
        out = [f'<grep_results chunk_count="{len(results)}">']
        for query in queries:
            out.append(f"<query>{xml_escape(query)}</query>")

        if not results:
            out.append("</grep_results>")
            return "\n".join(out)

        for item in results:
            chunk = item.chunk
            counts = count_regex_hits(chunk.content, compiled, queries)
            snippet = extract_chunk_match_snippet(chunk, compiled)
            faq_question = faq_standard_question(chunk)
            extra_attr = f' faq_question="{xml_escape(faq_question)}"' if faq_question else ""
            is_faq = chunk.chunk_type == CHUNK_TYPE_FAQ

            with self._seen_chunks_lock:
                seen = bool(self._seen_chunks.get(chunk.id))
                self._seen_chunks[chunk.id] = True
            seen_attr = ' already_seen="true"' if seen else ""

            if is_faq:
                out.append(
                    f'<faq faq_id="{xml_escape(chunk.id)}" '
                    f'knowledge_title="{xml_escape(item.knowledge_title)}"'
                    f"{extra_attr} index=\"{chunk.chunk_index}\" "
                    f'score="{item.match_score:.3f}"{seen_attr}>'
                )
            else:
                out.append(
                    f'<chunk chunk_id="{xml_escape(chunk.id)}" '
                    f'knowledge_id="{xml_escape(chunk.knowledge_id)}" '
                    f'knowledge_title="{xml_escape(item.knowledge_title)}"'
                    f"{extra_attr} chunk_index=\"{chunk.chunk_index}\" "
                    f'score="{item.match_score:.3f}"{seen_attr}>'
                )

            for query in queries:
                count = counts.get(query, 0)
                if count > 0:
                    out.append(
                        f'<query_hit query="{xml_escape(query)}" count="{count}" />'
                    )
            if seen:
                out.append(
                    "<note>(snippet omitted, already returned in a previous "
                    "grep_chunks call this session)</note>"
                )
            elif snippet:
                out.append(f"<match_snippet>{xml_escape(snippet)}</match_snippet>")

            out.append("</faq>" if is_faq else "</chunk>")

        out.append("</grep_results>")
        return "\n".join(out)

    def _aggregate_by_knowledge(
        self,
        results: Sequence[_ChunkWithTitle],
        queries: Sequence[str],
        compiled: Sequence[re.Pattern[str]],
    ) -> list[dict[str, Any]] | None:
        if not results:
            return None

        query_keys = [query for query in queries if query.strip()]
        aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}

        for item in results:
            chunk = item.chunk
            knowledge_key = (
                str(chunk.tenant_id),
                chunk.knowledge_base_id,
                chunk.knowledge_id or f"chunk-{chunk.id}",
            )
            if knowledge_key not in aggregated:
                title = item.knowledge_title.strip() or "Untitled"
                aggregated[knowledge_key] = {
                    "knowledge_id": knowledge_key[2],
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "knowledge_title": title,
                    "title_match": False,
                    "chunk_hit_count": 0,
                    "total_chunk_count": item.total_chunk_count,
                    "pattern_counts": {query: 0 for query in query_keys},
                    "total_pattern_hits": 0,
                    "distinct_patterns": 0,
                }

            entry = aggregated[knowledge_key]
            entry["chunk_hit_count"] = int(entry["chunk_hit_count"]) + 1
            if item.title_match:
                entry["title_match"] = True
            if not entry.get("faq_question"):
                question = faq_standard_question(chunk)
                if question:
                    entry["faq_question"] = question
            if not entry.get("match_snippet"):
                snippet = extract_chunk_match_snippet(chunk, compiled)
                if snippet:
                    entry["match_snippet"] = snippet

            occurrences = count_regex_hits(chunk.content, compiled, query_keys)
            pattern_counts = entry["pattern_counts"]
            for query in query_keys:
                count = occurrences.get(query, 0)
                if count <= 0:
                    continue
                pattern_counts[query] += count
                entry["total_pattern_hits"] = int(entry["total_pattern_hits"]) + count

        out = list(aggregated.values())
        for entry in out:
            entry["distinct_patterns"] = sum(
                1 for count in entry["pattern_counts"].values() if count > 0
            )

        out.sort(
            key=lambda entry: (
                not bool(entry["title_match"]),
                -int(entry["distinct_patterns"]),
                -int(entry["total_pattern_hits"]),
                -int(entry["chunk_hit_count"]),
                str(entry["knowledge_title"]),
            )
        )
        return out

    def _deduplicate_chunks(self, results: Sequence[_ChunkWithTitle]) -> list[_ChunkWithTitle]:
        seen: set[str] = set()
        content_sig: set[str] = set()
        unique_results: list[_ChunkWithTitle] = []

        for item in results:
            chunk = item.chunk
            keys = [chunk.id]
            if chunk.parent_chunk_id:
                keys.append("parent:" + chunk.parent_chunk_id)
            if chunk.knowledge_id:
                keys.append(
                    "scope:"
                    f"{chunk.tenant_id}:"
                    f"{chunk.knowledge_base_id}:"
                    f"{chunk.knowledge_id}#{chunk.chunk_index}"
                )

            if any(key in seen for key in keys):
                continue

            signature = build_content_signature(chunk.content)
            if signature:
                if signature in content_sig:
                    continue
                content_sig.add(signature)

            seen.update(keys)
            unique_results.append(item)

        seen_by_id: set[str] = set()
        deduplicated: list[_ChunkWithTitle] = []
        for item in unique_results:
            if item.chunk.id in seen_by_id:
                continue
            seen_by_id.add(item.chunk.id)
            deduplicated.append(item)
        return deduplicated

    def _score_chunks(
        self,
        results: Sequence[_ChunkWithTitle],
        compiled: Sequence[re.Pattern[str]],
    ) -> list[_ChunkWithTitle]:
        scored: list[_ChunkWithTitle] = []
        for item in results:
            score, pattern_count = self._calculate_match_score(item.chunk.content, compiled)
            title_match = regex_matches_any(item.knowledge_title, compiled)
            if title_match:
                score = min(score + 0.5, 1.0)
                if pattern_count == 0:
                    pattern_count = 1
            scored.append(
                _ChunkWithTitle(
                    chunk=item.chunk,
                    knowledge_title=item.knowledge_title,
                    match_score=score,
                    matched_patterns=pattern_count,
                    title_match=title_match,
                    total_chunk_count=item.total_chunk_count,
                )
            )
        return scored

    def _calculate_match_score(
        self,
        content: str,
        compiled: Sequence[re.Pattern[str]],
    ) -> tuple[float, int]:
        if content == "" or not compiled:
            return 0.0, 0

        match_count = 0
        earliest_pos = len(content)
        for pattern in compiled:
            match = pattern.search(content)
            if match is None:
                continue
            match_count += 1
            if match.start() < earliest_pos:
                earliest_pos = match.start()

        if match_count == 0:
            return 0.0, 0

        base_score = float(match_count) / float(len(compiled))
        position_bonus = 0.0
        if earliest_pos < len(content):
            position_ratio = 1.0 - float(earliest_pos) / float(len(content))
            position_bonus = position_ratio * 0.1
        return min(base_score + position_bonus, 1.0), match_count

    def _apply_mmr(
        self,
        results: Sequence[_ChunkWithTitle],
        k: int,
        lambda_value: float,
    ) -> list[_ChunkWithTitle]:
        if k <= 0 or not results:
            return []

        selected: list[_ChunkWithTitle] = []
        selected_token_sets: list[set[str]] = []
        candidates = list(results)
        token_sets = [tokenize_simple(item.chunk.content) for item in candidates]

        while len(selected) < k and candidates:
            best_idx = 0
            best_score = -1.0
            for index, item in enumerate(candidates):
                relevance = item.match_score
                redundancy = 0.0
                for selected_tokens in selected_token_sets:
                    redundancy = max(redundancy, jaccard(token_sets[index], selected_tokens))
                mmr_score = lambda_value * relevance - (1.0 - lambda_value) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = index

            selected.append(candidates[best_idx])
            selected_token_sets.append(token_sets[best_idx])

            last = len(candidates) - 1
            candidates[best_idx] = candidates[last]
            token_sets[best_idx] = token_sets[last]
            candidates = candidates[:last]
            token_sets = token_sets[:last]

        return selected


def _register_regexp(conn: sqlite3.Connection) -> None:
    def regexp(pattern: str, value: str | None) -> int:
        if value is None:
            return 0
        try:
            return 1 if re.search(str(pattern), str(value), re.IGNORECASE | re.MULTILINE) else 0
        except re.error:
            return 0

    conn.create_function("REGEXP", 2, regexp)


def _sync_temp_knowledge_titles(conn: sqlite3.Connection, titles: dict[str, str]) -> None:
    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS _tiny_rag_knowledge_titles (
            knowledge_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
        """
    )
    conn.execute("DELETE FROM _tiny_rag_knowledge_titles")
    rows = [(knowledge_id, title) for knowledge_id, title in titles.items() if knowledge_id]
    if rows:
        conn.executemany(
            "INSERT INTO _tiny_rag_knowledge_titles (knowledge_id, title) VALUES (?, ?)",
            rows,
        )


def build_chunk_scope_condition(target: SearchTarget) -> tuple[str, list[Any]]:
    if not target.knowledge_base_id:
        return "", []

    params: list[Any] = []
    conditions: list[str] = []
    knowledge_ids = [item for item in target.knowledge_ids if item]
    if target.target_type == "knowledge" or knowledge_ids:
        if not knowledge_ids:
            return "", []
        placeholders = ", ".join("?" for _ in knowledge_ids)
        conditions.append(f"chunks.knowledge_id IN ({placeholders})")
        params.extend(knowledge_ids)

    conditions.append("chunks.knowledge_base_id = ?")
    params.append(target.knowledge_base_id)
    conditions.append("chunks.tenant_id = ?")
    params.append(target.tenant_id)
    return "(" + " AND ".join(conditions) + ")", params


def build_grep_chunk_results(
    results: Sequence[_ChunkWithTitle],
    compiled: Sequence[re.Pattern[str]],
) -> list[dict[str, Any]] | None:
    if not results:
        return None
    out: list[dict[str, Any]] = []
    for item in results:
        chunk = item.chunk
        chunk_data: dict[str, Any] = {
            "knowledge_id": chunk.knowledge_id,
            "knowledge_base_id": chunk.knowledge_base_id,
            "knowledge_title": item.knowledge_title,
            "chunk_type": chunk.chunk_type,
            "title_match": item.title_match,
            "match_snippet": extract_chunk_match_snippet(chunk, compiled),
            "score": item.match_score,
        }
        if chunk.chunk_type == CHUNK_TYPE_FAQ:
            chunk_data["faq_id"] = chunk.id
            chunk_data["index"] = chunk.chunk_index
            question = faq_standard_question(chunk)
            if question:
                chunk_data["faq_question"] = question
        else:
            chunk_data["chunk_id"] = chunk.id
            chunk_data["chunk_index"] = chunk.chunk_index
        out.append(chunk_data)
    return out


def regex_matches_any(text: str, compiled: Sequence[re.Pattern[str]]) -> bool:
    if text == "" or not compiled:
        return False
    return any(pattern.search(text) is not None for pattern in compiled)


def count_regex_hits(
    content: str,
    compiled: Sequence[re.Pattern[str]],
    patterns: Sequence[str],
) -> dict[str, int]:
    counts = {pattern: 0 for pattern in patterns}
    if content == "" or not compiled:
        return counts
    for index, pattern in enumerate(compiled):
        if index >= len(patterns):
            continue
        counts[patterns[index]] = len(list(pattern.finditer(content)))
    return counts


def extract_chunk_match_snippet(
    chunk: ChunkRow | None,
    compiled: Sequence[re.Pattern[str]],
) -> str:
    if chunk is not None and chunk.chunk_type == CHUNK_TYPE_FAQ:
        snippet = faq_match_snippet(chunk, compiled)
        if snippet:
            return snippet
    if chunk is None:
        return ""
    return extract_snippet_regex(chunk.content, compiled)


def extract_snippet_regex(content: str, compiled: Sequence[re.Pattern[str]]) -> str:
    if content == "" or not compiled:
        return ""

    earliest = -1
    earliest_end = -1
    for pattern in compiled:
        match = pattern.search(content)
        if match is None:
            continue
        if earliest < 0 or match.start() < earliest:
            earliest = match.start()
            earliest_end = match.end()
    if earliest < 0:
        return ""

    match_text = content[earliest:earliest_end]
    before = content[:earliest]
    after = content[earliest_end:]

    before = before[-snippetContextRunes:]
    after = after[:snippetContextRunes]
    if len(match_text) > snippetMaxMatchRunes:
        match_text = match_text[:snippetMaxMatchRunes] + "..."

    snippet = before + match_text + after
    snippet = snippet.replace("\n", " ")
    while "  " in snippet:
        snippet = snippet.replace("  ", " ")
    snippet = snippet.strip()
    if len(snippet) > snippetMaxTotalRunes:
        snippet = snippet[:snippetMaxTotalRunes] + "..."
    return "... " + snippet + " ..."


def faq_standard_question(chunk: ChunkRow | None) -> str:
    if chunk is None or chunk.chunk_type != CHUNK_TYPE_FAQ:
        return ""
    meta = parse_faq_metadata(chunk.metadata)
    return str(meta.get("standard_question") or meta.get("StandardQuestion") or "").strip()


def faq_match_snippet(
    chunk: ChunkRow | None,
    compiled: Sequence[re.Pattern[str]],
) -> str:
    if chunk is None:
        return ""
    meta = parse_faq_metadata(chunk.metadata)
    if not meta:
        return ""
    question = faq_matched_question_from_regex(meta, compiled)
    if question == "":
        question = str(meta.get("standard_question") or meta.get("StandardQuestion") or "").strip()
    if question == "":
        return ""
    return format_faq_match_snippet(question, faq_answers(meta))


def faq_matched_question_from_regex(
    meta: dict[str, Any],
    compiled: Sequence[re.Pattern[str]],
) -> str:
    for question in faq_similar_questions(meta):
        if regex_matches_any(question, compiled):
            return question
    standard = str(meta.get("standard_question") or meta.get("StandardQuestion") or "").strip()
    if regex_matches_any(standard, compiled):
        return standard
    return standard


def format_faq_match_snippet(question: str, answers: Sequence[str]) -> str:
    question = question.strip()
    if question == "":
        return ""
    snippet = "Q: " + question
    answer = faq_answers_for_snippet(answers)
    if answer:
        snippet += " | A: " + answer
    snippet = snippet.strip()
    if len(snippet) > snippetMaxTotalRunes:
        snippet = truncate_runes(snippet, snippetMaxTotalRunes)
    return snippet


def faq_answers_for_snippet(answers: Sequence[str]) -> str:
    parts = [answer.strip() for answer in answers if answer.strip()]
    if not parts:
        return ""
    return truncate_runes(" | ".join(parts), snippetMaxAnswerRunes)


def faq_answers(meta: dict[str, Any]) -> list[str]:
    raw = meta.get("answers", meta.get("Answers", []))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Iterable):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def faq_similar_questions(meta: dict[str, Any]) -> list[str]:
    raw = meta.get("similar_questions", meta.get("SimilarQuestions", []))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Iterable):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def truncate_runes(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


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
        target_type = str(raw.get("type") or raw.get("Type") or "")
        if target_type == "":
            target_type = "knowledge" if knowledge_ids else "knowledge_base"
        return SearchTarget(
            knowledge_base_id=kb_id,
            knowledge_ids=knowledge_ids,
            tenant_id=int(raw.get("tenant_id") or raw.get("TenantID") or 0),
            target_type=target_type,
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
                target_type="knowledge",
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
