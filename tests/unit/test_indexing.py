from __future__ import annotations

import os
import threading
import time
from unittest import TestCase
from unittest.mock import patch

from tiny_rag.chunking import Chunk, Diagnostics, SplitterConfig
from tiny_rag.embedding.batch import batch_embed_with_backoff
from tiny_rag.embedding.sanitize import sanitize_for_embedding
from tiny_rag.indexing import (
    BatchSaveStats,
    IndexInfo,
    KEYWORDS_RETRIEVER_TYPE,
    KeywordsVectorHybridIndexer,
    VECTOR_RETRIEVER_TYPE,
    index_knowledge_after_chunks_persisted,
)
from tiny_rag.indexing.orchestrator import (
    DEFAULT_RETRIEVE_DRIVER,
    create_index_repository,
)
from tiny_rag.indexing.repositories.sqlite import SQLiteIndexRepository, tokenize_cjk_bigram
from tiny_rag.persistence import (
    ChunkRepository,
    ChunkRow,
    PersistedChunk,
    connect,
    persist_text_chunks,
)
from tiny_rag.persistence.chunk_models import CHUNK_TYPE_PARENT_TEXT, CHUNK_TYPE_TEXT


TENANT_ID = 7
KNOWLEDGE_ID = "knowledge-1"
KNOWLEDGE_BASE_ID = "kb-1"


class FakeEmbedder:
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        self.batch_calls: list[list[str]] = []
        self.text_to_index: dict[str, int] = {}

    @property
    def texts(self) -> list[str]:
        return [text for batch in self.batch_calls for text in batch]

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        embeddings: list[list[float]] = []
        for text in texts:
            if text not in self.text_to_index:
                self.text_to_index[text] = len(self.text_to_index) + 1
            embeddings.append([float(self.text_to_index[text])] * self.dimensions)
        return embeddings

    def get_model_name(self) -> str:
        return "fake"

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_model_id(self) -> str:
        return "fake"


class ExplodingEmbedder(FakeEmbedder):
    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embedder should not be called")


class FlakyOrderedEmbedder:
    def __init__(self):
        self.failures_remaining = 1
        self.batch_calls: list[list[str]] = []
        self.order = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("transient")
        return [[self.order[text]] for text in texts]

    def get_model_name(self) -> str:
        return "fake"

    def get_dimensions(self) -> int:
        return 1

    def get_model_id(self) -> str:
        return "fake"


class RecordingIndexRepository:
    def __init__(self, sleep_seconds: float = 0.0):
        self.sleep_seconds = sleep_seconds
        self.batch_sizes: list[int] = []
        self.max_active = 0
        self.active = 0
        self.lock = threading.Lock()

    def support(self) -> list[str]:
        return [KEYWORDS_RETRIEVER_TYPE, VECTOR_RETRIEVER_TYPE]

    def batch_save(self, index_info_list, *, embeddings_by_source_id=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            with self.lock:
                self.batch_sizes.append(len(index_info_list))
            vector_count = len(embeddings_by_source_id or {})
            return BatchSaveStats(
                metadata_count=len(index_info_list),
                fts_count=len(index_info_list),
                vector_count=vector_count,
                batches=1,
            )
        finally:
            with self.lock:
                self.active -= 1

    def delete_by_knowledge_id_list(self, knowledge_id_list, *, dimension=0, knowledge_type=""):
        return 0

    def estimate_storage_size(self, index_info_list, *, embeddings_by_source_id=None):
        return 0


class IndexRepositoryFactoryTests(TestCase):
    def test_default_retrieve_driver_creates_sqlite_repository(self):
        self.assertEqual(DEFAULT_RETRIEVE_DRIVER, "sqlite")

        with patch.dict(os.environ, {}, clear=True):
            repository = create_index_repository(connect(":memory:"))

        self.assertIsInstance(repository, SQLiteIndexRepository)

    def test_explicit_sqlite_driver_creates_sqlite_repository(self):
        with patch.dict(os.environ, {"RETRIEVE_DRIVER": "sqlite"}):
            repository = create_index_repository(connect(":memory:"))

        self.assertIsInstance(repository, SQLiteIndexRepository)

    def test_unsupported_retrieve_driver_raises(self):
        with patch.dict(os.environ, {"RETRIEVE_DRIVER": "postgres"}):
            with self.assertRaisesRegex(ValueError, "unsupported retrieve driver: postgres"):
                create_index_repository(connect(":memory:"))


class IndexingTests(TestCase):
    def make_conn(self):
        return connect(":memory:")

    def make_repo(self, conn):
        return SQLiteIndexRepository(conn)

    def create_row(
        self,
        conn,
        *,
        content: str,
        chunk_id: str,
        chunk_type: str = CHUNK_TYPE_TEXT,
        chunk_index: int = 0,
        knowledge_id: str = KNOWLEDGE_ID,
    ) -> ChunkRow:
        row = ChunkRow(
            id=chunk_id,
            tenant_id=TENANT_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            knowledge_id=knowledge_id,
            content=content,
            chunk_index=chunk_index,
            start_at=0,
            end_at=len(content),
            chunk_type=chunk_type,
        )
        ChunkRepository(conn).create_chunks([row])
        return row

    def index(
        self,
        conn,
        *,
        chunks,
        embedder=None,
        title: str = "",
        retriever_types=None,
    ):
        return index_knowledge_after_chunks_persisted(
            conn=conn,
            repository=self.make_repo(conn),
            embedder=embedder,
            knowledge_id=KNOWLEDGE_ID,
            title=title,
            chunks=chunks,
            retriever_types=retriever_types,
        )

    def test_heading_context_header_is_not_persisted_but_enters_embedding_input(self):
        conn = self.make_conn()
        row = self.create_row(conn, content="body", chunk_id="chunk-heading")
        embedder = FakeEmbedder()

        stats = self.index(
            conn,
            chunks=[PersistedChunk(row=row, context_header="# Heading")],
            embedder=embedder,
        )

        self.assertEqual(stats.embedded_count, 1)
        self.assertEqual(embedder.texts, ["# Heading\n\nbody"])
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        self.assertNotIn("context_header", columns)
        chunk_content = conn.execute(
            "SELECT content FROM chunks WHERE id = ?",
            ("chunk-heading",),
        ).fetchone()["content"]
        self.assertEqual(chunk_content, "body")

    def test_persist_result_inserted_chunks_preserve_heading_context_for_indexing(self):
        conn = self.make_conn()
        embedder = FakeEmbedder()
        diagnostics = Diagnostics(selected_tier="heading", tier_chain=["heading"])
        chunks = [Chunk(content="  body\n", context_header="# Heading", seq=0, start=0, end=7)]

        with patch(
            "tiny_rag.persistence.chunk_persist_service.split_with_diagnostics",
            return_value=(chunks, diagnostics),
        ):
            persist_result = persist_text_chunks(
                conn,
                tenant_id=TENANT_ID,
                knowledge_id=KNOWLEDGE_ID,
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                text="  body\n",
                config=SplitterConfig(strategy="heading"),
            )

        stats = self.index(
            conn,
            chunks=persist_result.inserted_chunks,
            embedder=embedder,
        )

        self.assertEqual(stats.embedded_count, 1)
        self.assertEqual(embedder.texts, ["# Heading\n\nbody"])
        chunk_content = conn.execute("SELECT content FROM chunks").fetchone()["content"]
        self.assertEqual(chunk_content, "  body\n")

    def test_post_persist_indexing_requires_in_memory_chunks_with_context_header(self):
        conn = self.make_conn()
        self.create_row(conn, content="body", chunk_id="chunk-db-only")

        with self.assertRaisesRegex(ValueError, "context_header"):
            index_knowledge_after_chunks_persisted(
                conn=conn,
                repository=self.make_repo(conn),
                embedder=FakeEmbedder(),
                knowledge_id=KNOWLEDGE_ID,
                chunks=None,
            )

    def test_parent_text_is_not_indexed_but_child_text_is_indexed(self):
        conn = self.make_conn()
        parent = self.create_row(
            conn,
            content="parent body",
            chunk_id="parent-1",
            chunk_type=CHUNK_TYPE_PARENT_TEXT,
            chunk_index=0,
        )
        child = self.create_row(conn, content="child body", chunk_id="child-1", chunk_index=1)
        child.parent_chunk_id = parent.id

        stats = self.index(
            conn,
            chunks=[PersistedChunk(row=parent), PersistedChunk(row=child)],
            retriever_types=[KEYWORDS_RETRIEVER_TYPE],
            embedder=ExplodingEmbedder(),
        )

        self.assertEqual(stats.text_chunk_count, 1)
        self.assertEqual(stats.metadata_count, 1)
        rows = conn.execute("SELECT chunk_id FROM lite_embeddings").fetchall()
        self.assertEqual([row["chunk_id"] for row in rows], ["child-1"])

    def test_title_prefix_is_first_in_embedding_input(self):
        conn = self.make_conn()
        row = self.create_row(conn, content="body", chunk_id="chunk-title")
        embedder = FakeEmbedder()

        self.index(
            conn,
            chunks=[PersistedChunk(row=row)],
            embedder=embedder,
            title="  Document Title  ",
        )

        self.assertEqual(embedder.texts, ["Document Title\nbody"])

    def test_chunk_db_content_is_original_but_embedding_content_is_trimmed(self):
        conn = self.make_conn()
        row = self.create_row(conn, content="\n  body  \n", chunk_id="chunk-trim")
        embedder = FakeEmbedder()

        self.index(conn, chunks=[PersistedChunk(row=row)], embedder=embedder)

        self.assertEqual(embedder.texts, ["body"])
        chunk_content = conn.execute(
            "SELECT content FROM chunks WHERE id = ?",
            ("chunk-trim",),
        ).fetchone()["content"]
        self.assertEqual(chunk_content, "\n  body  \n")
        lite_content = conn.execute("SELECT content FROM lite_embeddings").fetchone()["content"]
        self.assertEqual(lite_content, "body")

    def test_keyword_only_does_not_call_embedder_but_writes_metadata_and_fts(self):
        conn = self.make_conn()
        row = self.create_row(conn, content="hello keyword", chunk_id="chunk-keyword")

        stats = self.index(
            conn,
            chunks=[PersistedChunk(row=row)],
            retriever_types=[KEYWORDS_RETRIEVER_TYPE],
            embedder=ExplodingEmbedder(),
        )

        self.assertEqual(stats.embedded_count, 0)
        self.assertEqual(stats.metadata_count, 1)
        self.assertEqual(stats.fts_count, 1)
        self.assertEqual(stats.vector_count, 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings_fts").fetchone()[0], 1)
        vec_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'vec_embeddings_%'"
        ).fetchall()
        self.assertEqual(vec_tables, [])

    def test_vector_and_keyword_writes_metadata_fts_and_vec_table(self):
        conn = self.make_conn()
        row = self.create_row(conn, content="hello vector", chunk_id="chunk-vector")
        embedder = FakeEmbedder(dimensions=3)

        stats = self.index(
            conn,
            chunks=[PersistedChunk(row=row)],
            retriever_types=[KEYWORDS_RETRIEVER_TYPE, VECTOR_RETRIEVER_TYPE],
            embedder=embedder,
        )

        self.assertEqual(stats.embedded_count, 1)
        self.assertEqual(stats.metadata_count, 1)
        self.assertEqual(stats.fts_count, 1)
        self.assertEqual(stats.vector_count, 1)
        metadata = conn.execute(
            "SELECT source_id, source_type, chunk_id, knowledge_id, knowledge_base_id, dimension, is_enabled "
            "FROM lite_embeddings"
        ).fetchone()
        self.assertEqual(metadata["source_id"], "chunk-vector")
        self.assertEqual(metadata["source_type"], 0)
        self.assertEqual(metadata["chunk_id"], "chunk-vector")
        self.assertEqual(metadata["knowledge_id"], KNOWLEDGE_ID)
        self.assertEqual(metadata["knowledge_base_id"], KNOWLEDGE_BASE_ID)
        self.assertEqual(metadata["dimension"], 3)
        self.assertEqual(metadata["is_enabled"], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings_fts").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_embeddings_3").fetchone()[0], 1)

    def test_reindex_same_knowledge_deletes_old_index_before_writing_new_index(self):
        conn = self.make_conn()
        first = self.create_row(conn, content="old body", chunk_id="chunk-old")
        second = self.create_row(conn, content="new body", chunk_id="chunk-new", chunk_index=1)
        embedder = FakeEmbedder(dimensions=2)

        first_stats = self.index(conn, chunks=[PersistedChunk(row=first)], embedder=embedder)
        second_stats = self.index(conn, chunks=[PersistedChunk(row=second)], embedder=embedder)

        self.assertEqual(first_stats.deleted_count, 0)
        self.assertEqual(second_stats.deleted_count, 1)
        rows = conn.execute("SELECT chunk_id, content FROM lite_embeddings").fetchall()
        self.assertEqual([(row["chunk_id"], row["content"]) for row in rows], [("chunk-new", "new body")])
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings_fts").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_embeddings_2").fetchone()[0], 1)

    def test_batch_index_deduplicates_by_source_id_before_embedding_and_save(self):
        conn = self.make_conn()
        row = self.create_row(conn, content="same body", chunk_id="chunk-same")
        embedder = FakeEmbedder()

        stats = self.index(
            conn,
            chunks=[PersistedChunk(row=row), PersistedChunk(row=row)],
            embedder=embedder,
        )

        self.assertEqual(stats.requested_count, 2)
        self.assertEqual(stats.indexed_count, 1)
        self.assertEqual(stats.deduplicated_count, 1)
        self.assertEqual(embedder.texts, ["same body"])
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings").fetchone()[0], 1)

    def test_tokenize_cjk_bigram_matches_weknora_lite_behavior(self):
        self.assertEqual(tokenize_cjk_bigram("中国人 ABC，测试"), "中国 国人 ABC 测试")


class BatchEmbeddingTests(TestCase):
    @patch.dict(
        os.environ,
        {"BATCH_EMBED_SIZE": "2", "CONCURRENCY_POOL_SIZE": "1"},
    )
    @patch("tiny_rag.embedding.batch.time.sleep")
    def test_batch_embedding_uses_batch_size_preserves_order_and_retries(self, sleep_mock):
        embedder = FlakyOrderedEmbedder()

        result = batch_embed_with_backoff(embedder, ["a", "b", "c", "d", "e"])

        self.assertEqual(result, [[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.assertEqual(
            embedder.batch_calls,
            [["a", "b"], ["a", "b"], ["c", "d"], ["e"]],
        )
        sleep_mock.assert_called_once_with(0.2)


class BatchSaveServiceTests(TestCase):
    def index_infos(self, count: int) -> list[IndexInfo]:
        return [
            IndexInfo(
                content=f"content-{index}",
                source_id=f"source-{index}",
                chunk_id=f"chunk-{index}",
                knowledge_id=KNOWLEDGE_ID,
                knowledge_base_id=KNOWLEDGE_BASE_ID,
            )
            for index in range(count)
        ]

    @patch.dict(os.environ, {"BATCH_EMBED_SIZE": "100", "CONCURRENCY_POOL_SIZE": "1"})
    def test_vector_save_path_batches_by_40(self):
        repository = RecordingIndexRepository()
        indexer = KeywordsVectorHybridIndexer(repository)

        stats = indexer.batch_index(
            FakeEmbedder(dimensions=2),
            self.index_infos(45),
            [KEYWORDS_RETRIEVER_TYPE, VECTOR_RETRIEVER_TYPE],
        )

        self.assertCountEqual(repository.batch_sizes, [40, 5])
        self.assertEqual(stats.vector_count, 45)

    def test_keyword_only_save_path_batches_by_10_and_limits_concurrency_to_5(self):
        repository = RecordingIndexRepository(sleep_seconds=0.01)
        indexer = KeywordsVectorHybridIndexer(repository)

        stats = indexer.batch_index(
            ExplodingEmbedder(),
            self.index_infos(51),
            [KEYWORDS_RETRIEVER_TYPE],
        )

        self.assertCountEqual(repository.batch_sizes, [10, 10, 10, 10, 10, 1])
        self.assertLessEqual(repository.max_active, 5)
        self.assertEqual(stats.metadata_count, 51)
        self.assertEqual(stats.vector_count, 0)


class EmbeddingSanitizeTests(TestCase):
    def test_sanitize_removes_inline_base64_image_payload(self):
        payload = "A" * 300
        content = f'before <img src="data:image/png;base64,{payload}"> after'

        sanitized = sanitize_for_embedding(content)

        self.assertNotIn("data:image/png;base64", sanitized)
        self.assertNotIn(payload, sanitized)
        self.assertIn("before", sanitized)
        self.assertIn("after", sanitized)

    def test_sanitize_truncates_to_weknora_safety_max_runes(self):
        sanitized = sanitize_for_embedding("x" * 20001)

        self.assertEqual(len(sanitized), 20000)
