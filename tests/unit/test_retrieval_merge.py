from __future__ import annotations

import json
from unittest import TestCase

from tiny_rag.persistence import ChunkRepository, ChunkRow, connect
from tiny_rag.persistence.chunk_models import CHUNK_TYPE_PARENT_TEXT, CHUNK_TYPE_TEXT
from tiny_rag.retrieval import CHUNK_TYPE_FAQ, CHUNK_TYPE_TEXT as RETRIEVAL_CHUNK_TYPE_TEXT
from tiny_rag.retrieval.merge import MergeService
from tiny_rag.retrieval.models import SearchResult


TENANT_ID = 7
KNOWLEDGE_ID = "knowledge-1"
KNOWLEDGE_BASE_ID = "kb-1"


def search_result(chunk_id: str, content: str, *, score: float = 0.5, **kwargs):
    return SearchResult(
        id=chunk_id,
        content=content,
        knowledge_id=kwargs.pop("knowledge_id", KNOWLEDGE_ID),
        knowledge_base_id=kwargs.pop("knowledge_base_id", KNOWLEDGE_BASE_ID),
        score=score,
        start_at=kwargs.pop("start_at", 0),
        end_at=kwargs.pop("end_at", len(content)),
        chunk_type=kwargs.pop("chunk_type", RETRIEVAL_CHUNK_TYPE_TEXT),
        **kwargs,
    )


class MergeServiceTests(TestCase):
    def make_repo(self):
        conn = connect(":memory:")
        return conn, ChunkRepository(conn)

    def add_chunk(self, repo, chunk_id: str, content: str, **kwargs):
        row = ChunkRow(
            id=chunk_id,
            tenant_id=kwargs.pop("tenant_id", TENANT_ID),
            knowledge_base_id=kwargs.pop("knowledge_base_id", KNOWLEDGE_BASE_ID),
            knowledge_id=kwargs.pop("knowledge_id", KNOWLEDGE_ID),
            content=content,
            chunk_index=kwargs.pop("chunk_index", 0),
            start_at=kwargs.pop("start_at", 0),
            end_at=kwargs.pop("end_at", len(content)),
            chunk_type=kwargs.pop("chunk_type", CHUNK_TYPE_TEXT),
            pre_chunk_id=kwargs.pop("pre_chunk_id", ""),
            next_chunk_id=kwargs.pop("next_chunk_id", ""),
            parent_chunk_id=kwargs.pop("parent_chunk_id", ""),
            metadata=kwargs.pop("metadata", None),
        )
        repo.create_chunks([row])
        return row

    def test_dedup_by_chunk_id_and_content_signature(self):
        service = MergeService()
        results = [
            search_result("a", "same content", score=0.9),
            search_result("a", "different content", score=0.8),
            search_result("b", "  Same   Content  ", score=0.7),
            search_result("c", "unique content", score=0.6, start_at=100, end_at=114),
        ]

        merged = service.merge(results)

        self.assertEqual([item.id for item in merged], ["a", "c"])

    def test_empty_rerank_results_fall_back_to_search_results_sorted_by_score(self):
        service = MergeService()
        search_results = [
            search_result("low", "low content", score=0.1, knowledge_id="k-low"),
            search_result("high", "high content", score=0.9, knowledge_id="k-high"),
        ]

        merged = service.merge(search_results=search_results, rerank_results=[])

        self.assertEqual([item.id for item in merged], ["high", "low"])

    def test_overlap_merge_appends_non_overlapping_suffix(self):
        service = MergeService()
        first = search_result("a", "abcdefghijklmnopqrstuvwxyz", score=0.5, start_at=0, end_at=26)
        second = search_result("b", "opqrstuvwxyz123", score=0.8, start_at=14, end_at=29)

        merged = service.merge([first, second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].content, "abcdefghijklmnopqrstuvwxyz123")
        self.assertEqual(merged[0].score, 0.8)
        self.assertIn("b", merged[0].sub_chunk_id)

    def test_parent_resolve_expands_text_child_to_parent_content(self):
        _, repo = self.make_repo()
        parent_content = "P" * 420
        self.add_chunk(
            repo,
            "parent",
            parent_content,
            chunk_type=CHUNK_TYPE_PARENT_TEXT,
            start_at=0,
            end_at=len(parent_content),
        )
        self.add_chunk(
            repo,
            "child",
            parent_content[100:150],
            parent_chunk_id="parent",
            start_at=100,
            end_at=150,
        )
        service = MergeService(repo, tenant_id=TENANT_ID)
        result = search_result(
            "child",
            parent_content[100:150],
            parent_chunk_id="parent",
            start_at=100,
            end_at=150,
        )

        merged = service.merge([result])

        self.assertEqual(merged[0].content, parent_content)
        self.assertEqual(merged[0].start_at, 0)
        self.assertEqual(merged[0].end_at, len(parent_content))
        self.assertIn("child", merged[0].sub_chunk_id)

    def test_neighbor_expansion_for_short_text_context(self):
        _, repo = self.make_repo()
        prev_content = "A" * 180
        base_content = "B" * 40
        next_content = "C" * 180
        self.add_chunk(
            repo,
            "prev",
            prev_content,
            chunk_index=0,
            start_at=0,
            end_at=180,
            next_chunk_id="base",
        )
        self.add_chunk(
            repo,
            "base",
            base_content,
            chunk_index=1,
            start_at=180,
            end_at=220,
            pre_chunk_id="prev",
            next_chunk_id="next",
        )
        self.add_chunk(
            repo,
            "next",
            next_content,
            chunk_index=2,
            start_at=220,
            end_at=400,
            pre_chunk_id="base",
        )
        service = MergeService(repo, tenant_id=TENANT_ID)

        merged = service.merge([search_result("base", base_content, start_at=180, end_at=220)])

        self.assertEqual(merged[0].content, prev_content + base_content + next_content)
        self.assertIn("prev", merged[0].sub_chunk_id)
        self.assertIn("next", merged[0].sub_chunk_id)

    def test_faq_populate_uses_answers_from_chunk_metadata(self):
        _, repo = self.make_repo()
        metadata = json.dumps(
            {
                "standard_question": "What is Tiny RAG?",
                "answers": ["A small retrieval stack.", "It uses SQLite."],
            }
        )
        self.add_chunk(
            repo,
            "faq",
            "question only",
            chunk_type=CHUNK_TYPE_FAQ,
            metadata=metadata,
        )
        service = MergeService(repo, tenant_id=TENANT_ID)
        item = search_result("faq", "question only", chunk_type=CHUNK_TYPE_FAQ)

        merged = service.merge([item])

        self.assertEqual(
            merged[0].content,
            "Q: What is Tiny RAG?\nAnswer:\n- A small retrieval stack.\n- It uses SQLite.",
        )
