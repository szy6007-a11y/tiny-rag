from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

from tiny_rag.persistence import connect
from tiny_rag.persistence.sqlite import transaction
from tiny_rag.service import (
    IngestRequest,
    IngestService,
    KnowledgeBaseConfig,
    KnowledgeRecord,
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_PENDING,
)


class FakeEmbedder:
    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]

    def get_model_name(self) -> str:
        return "fake"

    def get_model_id(self) -> str:
        return "fake"

    def get_dimensions(self) -> int:
        return 2


class IngestLifecycleTests(TestCase):
    def make_service(self):
        conn = connect(":memory:")
        self.addCleanup(conn.close)
        kb = KnowledgeBaseConfig(id="kb-life", tenant_id=7, embedding_model_id="fake")
        service = IngestService(conn, embedder=FakeEmbedder(), knowledge_bases={kb.id: kb})
        return conn, kb, service

    def ingest(self, service: IngestService, kb: KnowledgeBaseConfig, text: str):
        return service.ingest_document(
            IngestRequest(
                tenant_id=7,
                knowledge_id="knowledge-life",
                knowledge_base_id=kb.id,
                content=text,
                title="Lifecycle",
                file_name="life.md",
                file_type="md",
                source="/docs/life.md",
                knowledge_base=kb,
            )
        )

    def test_hash_unchanged_skips_and_changed_source_replaces_same_knowledge(self):
        conn, kb, service = self.make_service()

        first = self.ingest(service, kb, "alpha beta")
        second = self.ingest(service, kb, "alpha beta")
        third = self.ingest(service, kb, "gamma delta")

        self.assertEqual(first.operation, "created")
        self.assertTrue(second.skipped)
        self.assertEqual(second.message, "unchanged")
        self.assertEqual(third.operation, "updated")
        self.assertEqual(third.knowledge.id, "knowledge-life")
        self.assertEqual(third.knowledge.parse_status, PARSE_STATUS_COMPLETED)
        self.assertEqual(third.index_stats.deleted_count, 1)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM chunks WHERE deleted_at IS NULL").fetchone()[0],
            1,
        )
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings").fetchone()[0], 1)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM knowledges WHERE deleted_at IS NULL").fetchone()[0],
            1,
        )

    def test_delete_marks_knowledge_and_cleans_chunks_and_index(self):
        conn, kb, service = self.make_service()
        self.ingest(service, kb, "alpha beta")

        deleted = service.delete_knowledge(tenant_id=7, knowledge_id="knowledge-life")

        self.assertEqual(deleted.id, "knowledge-life")
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM chunks WHERE deleted_at IS NULL").fetchone()[0],
            0,
        )
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings").fetchone()[0], 0)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM knowledges WHERE deleted_at IS NULL").fetchone()[0],
            0,
        )
        row = conn.execute("SELECT parse_status, deleted_at FROM knowledges").fetchone()
        self.assertEqual(row["parse_status"], "deleting")
        self.assertIsNotNone(row["deleted_at"])

    def test_cancelled_knowledge_short_circuits_processing(self):
        conn, kb, service = self.make_service()
        pending = KnowledgeRecord(
            id="pending-life",
            tenant_id=7,
            knowledge_base_id=kb.id,
            title="Pending",
            parse_status=PARSE_STATUS_PENDING,
        )
        with transaction(conn):
            service.knowledge_repository.upsert(pending)

        cancelled = service.cancel_knowledge_parse(tenant_id=7, knowledge_id="pending-life")
        result = service.process_document(
            knowledge=cancelled,
            knowledge_base=kb,
            payload=b"content that should not be parsed",
            file_name="pending.md",
            file_type="md",
        )

        self.assertEqual(cancelled.parse_status, PARSE_STATUS_CANCELLED)
        self.assertTrue(result.skipped)
        self.assertEqual(result.message, PARSE_STATUS_CANCELLED)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings").fetchone()[0], 0)

    def test_reparse_uses_stored_file_path_and_same_knowledge_id(self):
        conn, kb, service = self.make_service()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual.md"
            path.write_text("# Manual\n\nalpha beta", encoding="utf-8")

            first = service.ingest_document(
                IngestRequest(
                    tenant_id=7,
                    knowledge_id="knowledge-file",
                    knowledge_base_id=kb.id,
                    content=path,
                    title="Manual",
                    knowledge_base=kb,
                )
            )
            path.write_text("# Manual\n\ngamma delta", encoding="utf-8")

            reparsed = service.reparse_knowledge(tenant_id=7, knowledge_id=first.knowledge.id)

        self.assertEqual(reparsed.knowledge.id, "knowledge-file")
        self.assertEqual(reparsed.operation, "updated")
        active_content = [
            row["content"]
            for row in conn.execute(
                "SELECT content FROM chunks WHERE deleted_at IS NULL ORDER BY chunk_index"
            )
        ]
        self.assertEqual(active_content, ["# Manual\n\ngamma delta"])
