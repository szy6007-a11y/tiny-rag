from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from tiny_rag.indexing.models import (
    CHUNK_SOURCE_TYPE,
    IndexInfo,
    KEYWORDS_RETRIEVER_TYPE,
    MATCH_TYPE_EMBEDDING,
    MATCH_TYPE_KEYWORDS,
    RetrieveParams,
    SQLITE_RETRIEVER_ENGINE_TYPE,
    VECTOR_RETRIEVER_TYPE,
)
from tiny_rag.indexing.repositories.sqlite import (
    SQLiteIndexRepository,
    connect_sqlite_index_db,
)


class SQLiteIndexRepositoryTests(TestCase):
    def make_repo(self):
        conn = connect_sqlite_index_db(":memory:")
        return conn, SQLiteIndexRepository(conn)

    def info(
        self,
        suffix: str,
        *,
        content: str | None = None,
        knowledge_id: str = "knowledge-1",
        knowledge_base_id: str = "kb-1",
        tag_id: str = "",
        is_enabled: bool = True,
    ) -> IndexInfo:
        return IndexInfo(
            content=content or f"alpha keyword {suffix}",
            source_id=f"source-{suffix}",
            source_type=CHUNK_SOURCE_TYPE,
            chunk_id=f"chunk-{suffix}",
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            tag_id=tag_id,
            is_enabled=is_enabled,
        )

    def save_vector(self, repo, suffix: str, **kwargs) -> IndexInfo:
        info = self.info(suffix, **kwargs)
        repo.save(info, embeddings_by_source_id={info.source_id: self.vector_for(suffix)})
        return info

    def vector_for(self, suffix: str) -> list[float]:
        if suffix.endswith("1"):
            return [1.0, 0.0]
        if suffix.endswith("2"):
            return [0.0, 1.0]
        return [0.5, 0.5]

    def counts(self, conn, dim: int = 2) -> tuple[int, int, int]:
        metadata_count = conn.execute("SELECT COUNT(*) FROM lite_embeddings").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM lite_embeddings_fts").fetchone()[0]
        vec_count = conn.execute(f"SELECT COUNT(*) FROM vec_embeddings_{dim}").fetchone()[0]
        return metadata_count, fts_count, vec_count

    def test_save_and_batch_save_write_same_metadata_fts_and_vec_paths(self):
        conn, repo = self.make_repo()
        single = self.info("1")
        batch = self.info("2")

        single_stats = repo.save(
            single,
            embeddings_by_source_id={single.source_id: [1.0, 0.0]},
        )
        batch_stats = repo.batch_save(
            [batch],
            embeddings_by_source_id={batch.source_id: [0.0, 1.0]},
        )

        self.assertEqual(single_stats.metadata_count, 1)
        self.assertEqual(single_stats.fts_count, 1)
        self.assertEqual(single_stats.vector_count, 1)
        self.assertEqual(batch_stats.metadata_count, 1)
        self.assertEqual(batch_stats.fts_count, 1)
        self.assertEqual(batch_stats.vector_count, 1)
        self.assertEqual(self.counts(conn), (2, 2, 2))

        rows = conn.execute(
            """
            SELECT source_id, source_type, chunk_id, knowledge_id, knowledge_base_id, content, dimension
            FROM lite_embeddings
            ORDER BY chunk_id
            """
        ).fetchall()
        self.assertEqual([row["source_id"] for row in rows], ["source-1", "source-2"])
        self.assertEqual([row["source_type"] for row in rows], [0, 0])
        self.assertEqual([row["chunk_id"] for row in rows], ["chunk-1", "chunk-2"])
        self.assertEqual([row["knowledge_id"] for row in rows], ["knowledge-1", "knowledge-1"])
        self.assertEqual([row["knowledge_base_id"] for row in rows], ["kb-1", "kb-1"])
        self.assertEqual([row["dimension"] for row in rows], [2, 2])

    def test_source_and_type_conflict_does_nothing(self):
        conn, repo = self.make_repo()
        info = self.save_vector(repo, "1")

        stats = repo.batch_save(
            [info],
            embeddings_by_source_id={info.source_id: [1.0, 0.0]},
        )

        self.assertEqual(stats.metadata_count, 0)
        self.assertEqual(stats.fts_count, 0)
        self.assertEqual(stats.vector_count, 0)
        self.assertEqual(stats.skipped_duplicate_count, 1)
        self.assertEqual(self.counts(conn), (1, 1, 1))

    def test_estimate_storage_size_matches_weknora_sqlite(self):
        _, repo = self.make_repo()
        infos = [
            IndexInfo(content="abc", source_id="s1"),
            IndexInfo(content="中文", source_id="s2"),
        ]

        self.assertEqual(repo.estimate_storage_size(infos), len("abc") + len("中文".encode("utf-8")) + 400)

    def test_delete_by_chunk_id_list_syncs_metadata_fts_and_vec(self):
        conn, repo = self.make_repo()
        self.save_vector(repo, "1")
        self.save_vector(repo, "2")

        deleted = repo.delete_by_chunk_id_list(["chunk-1"], dimension=2, knowledge_type="manual")

        self.assertEqual(deleted, 1)
        self.assertEqual(self.counts(conn), (1, 1, 1))
        remaining = conn.execute("SELECT chunk_id FROM lite_embeddings").fetchone()["chunk_id"]
        self.assertEqual(remaining, "chunk-2")

    def test_delete_by_source_id_list_syncs_metadata_fts_and_vec(self):
        conn, repo = self.make_repo()
        self.save_vector(repo, "1")
        self.save_vector(repo, "2")

        deleted = repo.delete_by_source_id_list(["source-2"], dimension=2, knowledge_type="manual")

        self.assertEqual(deleted, 1)
        self.assertEqual(self.counts(conn), (1, 1, 1))
        remaining = conn.execute("SELECT source_id FROM lite_embeddings").fetchone()["source_id"]
        self.assertEqual(remaining, "source-1")

    def test_delete_by_knowledge_id_list_syncs_metadata_fts_and_vec(self):
        conn, repo = self.make_repo()
        self.save_vector(repo, "1", knowledge_id="knowledge-1")
        self.save_vector(repo, "2", knowledge_id="knowledge-2")

        deleted = repo.delete_by_knowledge_id_list(["knowledge-1"], dimension=2, knowledge_type="manual")

        self.assertEqual(deleted, 1)
        self.assertEqual(self.counts(conn), (1, 1, 1))
        remaining = conn.execute("SELECT knowledge_id FROM lite_embeddings").fetchone()["knowledge_id"]
        self.assertEqual(remaining, "knowledge-2")

    def test_copy_indices_copies_metadata_fts_and_vec_with_new_source_id(self):
        conn, repo = self.make_repo()
        source = self.save_vector(
            repo,
            "1",
            content="copy alpha",
            knowledge_id="source-knowledge",
            knowledge_base_id="source-kb",
            tag_id="tag-1",
            is_enabled=False,
        )

        repo.copy_indices(
            "source-kb",
            {"source-knowledge": "target-knowledge"},
            {source.chunk_id: "target-chunk"},
            "target-kb",
            dimension=2,
            knowledge_type="manual",
        )

        rows = conn.execute(
            """
            SELECT source_id, source_type, chunk_id, knowledge_id, knowledge_base_id,
                   tag_id, content, dimension, is_enabled
            FROM lite_embeddings
            ORDER BY id
            """
        ).fetchall()
        self.assertEqual(len(rows), 2)
        copied = rows[1]
        self.assertNotEqual(copied["source_id"], rows[0]["source_id"])
        self.assertEqual(copied["source_type"], rows[0]["source_type"])
        self.assertEqual(copied["chunk_id"], "target-chunk")
        self.assertEqual(copied["knowledge_id"], "target-knowledge")
        self.assertEqual(copied["knowledge_base_id"], "target-kb")
        self.assertEqual(copied["tag_id"], "tag-1")
        self.assertEqual(copied["content"], "copy alpha")
        self.assertEqual(copied["dimension"], 2)
        self.assertEqual(copied["is_enabled"], 0)
        self.assertEqual(self.counts(conn), (2, 2, 2))

    def test_copy_indices_uses_lowest_id_source_when_chunk_id_matches_multiple_rows(self):
        conn, repo = self.make_repo()
        source = self.save_vector(
            repo,
            "1",
            content="first source",
            knowledge_id="source-knowledge",
        )
        duplicate = IndexInfo(
            content="second source",
            source_id="source-duplicate",
            source_type=CHUNK_SOURCE_TYPE,
            chunk_id=source.chunk_id,
            knowledge_id="source-knowledge",
            knowledge_base_id="source-kb",
        )
        repo.save(duplicate, embeddings_by_source_id={duplicate.source_id: [0.0, 1.0]})

        repo.copy_indices(
            "source-kb",
            {"source-knowledge": "target-knowledge"},
            {source.chunk_id: "target-chunk"},
            "target-kb",
        )

        copied = conn.execute(
            "SELECT content FROM lite_embeddings WHERE chunk_id = ?",
            ("target-chunk",),
        ).fetchone()
        self.assertEqual(copied["content"], "first source")

    def test_copy_indices_continues_after_single_insert_failure(self):
        conn, repo = self.make_repo()
        first = self.save_vector(repo, "1", knowledge_id="source-knowledge")
        second = self.save_vector(repo, "2", knowledge_id="source-knowledge")
        conflicting_source_id = "copy-source-conflict"
        repo.save(
            IndexInfo(
                content="preexisting conflict",
                source_id=conflicting_source_id,
                source_type=CHUNK_SOURCE_TYPE,
                chunk_id="preexisting-conflict",
                knowledge_id="other-knowledge",
                knowledge_base_id="other-kb",
            )
        )

        with patch(
            "tiny_rag.indexing.repositories.sqlite.uuid4",
            side_effect=[conflicting_source_id, "copy-source-success"],
        ):
            repo.copy_indices(
                "source-kb",
                {"source-knowledge": "target-knowledge"},
                {
                    first.chunk_id: "target-failed",
                    second.chunk_id: "target-success",
                },
                "target-kb",
            )

        copied_rows = conn.execute(
            """
            SELECT source_id, chunk_id, knowledge_id, knowledge_base_id
            FROM lite_embeddings
            WHERE knowledge_base_id = 'target-kb'
            ORDER BY chunk_id
            """
        ).fetchall()
        self.assertEqual(len(copied_rows), 1)
        self.assertEqual(copied_rows[0]["source_id"], "copy-source-success")
        self.assertEqual(copied_rows[0]["chunk_id"], "target-success")
        self.assertEqual(copied_rows[0]["knowledge_id"], "target-knowledge")
        self.assertEqual(self.counts(conn), (4, 4, 3))

    def test_copy_indices_skips_missing_source_chunk(self):
        conn, repo = self.make_repo()

        repo.copy_indices(
            "source-kb",
            {"source-knowledge": "target-knowledge"},
            {"missing-chunk": "target-chunk"},
            "target-kb",
        )

        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM lite_embeddings_fts").fetchone()[0], 0)

    def test_batch_update_chunk_enabled_status_updates_by_chunk_id(self):
        conn, repo = self.make_repo()
        self.save_vector(repo, "1", is_enabled=True)
        self.save_vector(repo, "2", is_enabled=False)

        repo.batch_update_chunk_enabled_status({"chunk-1": False, "chunk-2": True})

        rows = conn.execute(
            "SELECT chunk_id, is_enabled FROM lite_embeddings ORDER BY chunk_id"
        ).fetchall()
        self.assertEqual([(row["chunk_id"], row["is_enabled"]) for row in rows], [("chunk-1", 0), ("chunk-2", 1)])

    def test_batch_update_chunk_tag_id_updates_by_chunk_id(self):
        conn, repo = self.make_repo()
        self.save_vector(repo, "1", tag_id="")
        self.save_vector(repo, "2", tag_id="old")

        repo.batch_update_chunk_tag_id({"chunk-1": "tag-a", "chunk-2": ""})

        rows = conn.execute("SELECT chunk_id, tag_id FROM lite_embeddings ORDER BY chunk_id").fetchall()
        self.assertEqual([(row["chunk_id"], row["tag_id"]) for row in rows], [("chunk-1", "tag-a"), ("chunk-2", "")])

    def test_retrieve_keyword_query_returns_result(self):
        _, repo = self.make_repo()
        self.save_vector(repo, "1", content="alpha apple banana")
        self.save_vector(repo, "2", content="orange pear")

        results = repo.retrieve(
            RetrieveParams(query="apple", top_k=5, retriever_type=KEYWORDS_RETRIEVER_TYPE)
        )

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].error)
        self.assertEqual(results[0].retriever_engine_type, SQLITE_RETRIEVER_ENGINE_TYPE)
        self.assertEqual(results[0].retriever_type, KEYWORDS_RETRIEVER_TYPE)
        self.assertEqual([item.chunk_id for item in results[0].results], ["chunk-1"])
        self.assertEqual(results[0].results[0].match_type, MATCH_TYPE_KEYWORDS)

    def test_retrieve_keyword_applies_enabled_and_metadata_filters(self):
        _, repo = self.make_repo()
        self.save_vector(repo, "1", content="alpha apple", knowledge_id="k1", knowledge_base_id="kb1", tag_id="t1")
        self.save_vector(repo, "2", content="alpha apple", knowledge_id="k2", knowledge_base_id="kb2", tag_id="t2")
        repo.batch_update_chunk_enabled_status({"chunk-2": False})

        results = repo.retrieve(
            RetrieveParams(
                query="apple",
                top_k=5,
                retriever_type=KEYWORDS_RETRIEVER_TYPE,
                knowledge_base_ids=["kb1"],
                knowledge_ids=["k1"],
                tag_ids=["t1"],
            )
        )

        self.assertEqual([item.chunk_id for item in results[0].results], ["chunk-1"])

    def test_retrieve_vector_query_returns_result(self):
        _, repo = self.make_repo()
        self.save_vector(repo, "1", content="alpha vector")
        self.save_vector(repo, "2", content="beta vector")

        results = repo.retrieve(
            RetrieveParams(embedding=[1.0, 0.0], top_k=2, retriever_type=VECTOR_RETRIEVER_TYPE)
        )

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].error)
        self.assertEqual(results[0].retriever_engine_type, SQLITE_RETRIEVER_ENGINE_TYPE)
        self.assertEqual(results[0].retriever_type, VECTOR_RETRIEVER_TYPE)
        self.assertEqual(results[0].results[0].chunk_id, "chunk-1")
        self.assertEqual(results[0].results[0].match_type, MATCH_TYPE_EMBEDDING)
        self.assertGreaterEqual(results[0].results[0].score, results[0].results[1].score)

    def test_retrieve_empty_retriever_type_runs_keyword_and_vector_paths(self):
        _, repo = self.make_repo()
        self.save_vector(repo, "1", content="alpha apple")
        self.save_vector(repo, "2", content="beta orange")

        results = repo.retrieve(RetrieveParams(query="apple", embedding=[1.0, 0.0], top_k=2))

        self.assertEqual([result.retriever_type for result in results], [KEYWORDS_RETRIEVER_TYPE, VECTOR_RETRIEVER_TYPE])
        self.assertEqual(results[0].results[0].chunk_id, "chunk-1")
        self.assertEqual(results[1].results[0].chunk_id, "chunk-1")

    def test_retrieve_empty_query_or_embedding_skips_that_path(self):
        _, repo = self.make_repo()
        self.save_vector(repo, "1", content="alpha apple")

        self.assertEqual(repo.retrieve(RetrieveParams(query="", top_k=5, retriever_type=KEYWORDS_RETRIEVER_TYPE)), [])
        self.assertEqual(repo.retrieve(RetrieveParams(embedding=[], top_k=5, retriever_type=VECTOR_RETRIEVER_TYPE)), [])
