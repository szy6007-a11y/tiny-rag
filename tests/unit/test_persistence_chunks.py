from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from tiny_rag.chunking import ChildChunk, Chunk, Diagnostics, ParentChildResult, SplitterConfig
from tiny_rag.persistence import ChunkRepository, connect, persist_text_chunks


TENANT_ID = 7
KNOWLEDGE_ID = "knowledge-1"
KNOWLEDGE_BASE_ID = "kb-1"


def normal_text() -> str:
    return ("alpha beta gamma delta epsilon\n\n" * 8).strip() + "\n"


def normal_config() -> SplitterConfig:
    return SplitterConfig(chunk_size=70, chunk_overlap=1, strategy="legacy")


class PersistChunksTests(TestCase):
    def make_conn(self):
        return connect(":memory:")

    def active_rows(self, conn, knowledge_id: str = KNOWLEDGE_ID):
        return conn.execute(
            """
            SELECT *
            FROM chunks
            WHERE tenant_id = ?
              AND knowledge_id = ?
              AND deleted_at IS NULL
            ORDER BY chunk_index ASC, seq_id ASC
            """,
            (TENANT_ID, knowledge_id),
        ).fetchall()

    def test_flat_persist_preserves_original_content_slice(self):
        conn = self.make_conn()
        text = normal_text()

        result = persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=text,
            config=normal_config(),
        )

        self.assertGreater(result.inserted_count, 1)
        for chunk in result.inserted_chunks:
            self.assertEqual(chunk.content, text[chunk.start_at : chunk.end_at])

        for row in self.active_rows(conn):
            self.assertEqual(row["content"], text[row["start_at"] : row["end_at"]])

    def test_flat_chunk_index_order_is_ascending(self):
        conn = self.make_conn()

        result = persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=normal_text(),
            config=normal_config(),
        )

        indexes = [chunk.chunk_index for chunk in result.inserted_chunks]
        self.assertEqual(indexes, sorted(indexes))
        self.assertEqual(indexes, list(range(len(indexes))))

    def test_flat_prev_next_links_are_bidirectional(self):
        conn = self.make_conn()
        persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=normal_text(),
            config=normal_config(),
        )

        rows = self.active_rows(conn)
        self.assertGreater(len(rows), 1)
        for index, row in enumerate(rows):
            expected_prev = "" if index == 0 else rows[index - 1]["id"]
            expected_next = "" if index == len(rows) - 1 else rows[index + 1]["id"]
            self.assertEqual(row["pre_chunk_id"], expected_prev)
            self.assertEqual(row["next_chunk_id"], expected_next)
            if expected_prev:
                self.assertEqual(rows[index - 1]["next_chunk_id"], row["id"])
            if expected_next:
                self.assertEqual(rows[index + 1]["pre_chunk_id"], row["id"])

    def test_weknora_default_chunk_fields_are_written(self):
        conn = self.make_conn()
        persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=normal_text(),
            config=normal_config(),
        )

        row = self.active_rows(conn)[0]
        self.assertEqual(row["is_enabled"], 1)
        self.assertEqual(row["flags"], 0)
        self.assertEqual(row["status"], 0)
        self.assertEqual(row["tag_id"], "")
        self.assertEqual(row["metadata"], None)
        self.assertEqual(row["content_hash"], "")
        self.assertEqual(row["image_info"], "")
        self.assertEqual(row["video_info"], "")

    def test_parent_child_writes_parent_and_child_rows(self):
        conn = self.make_conn()
        text = ("alpha beta gamma delta epsilon\n\n" * 20).strip() + "\n"

        result = persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=text,
            config=SplitterConfig(
                parent_child=True,
                parent_chunk_size=1000,
                child_chunk_size=60,
                chunk_overlap=1,
                strategy="legacy",
            ),
        )

        rows = self.active_rows(conn)
        parents = [row for row in rows if row["chunk_type"] == "parent_text"]
        children = [row for row in rows if row["chunk_type"] == "text"]
        self.assertEqual(result.parent_count, 1)
        self.assertEqual(len(parents), 1)
        self.assertGreater(len(children), 1)
        self.assertEqual(parents[0]["content"], text[parents[0]["start_at"] : parents[0]["end_at"]])
        for child in children:
            self.assertEqual(child["parent_chunk_id"], parents[0]["id"])
            self.assertEqual(child["content"], text[child["start_at"] : child["end_at"]])

    def test_parent_child_children_do_not_write_prev_next(self):
        conn = self.make_conn()
        text = ("alpha beta gamma delta epsilon\n\n" * 20).strip() + "\n"

        persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=text,
            config=SplitterConfig(
                parent_child=True,
                parent_chunk_size=1000,
                child_chunk_size=60,
                chunk_overlap=1,
                strategy="legacy",
            ),
        )

        children = [row for row in self.active_rows(conn) if row["chunk_type"] == "text"]
        self.assertTrue(children)
        for child in children:
            self.assertEqual(child["pre_chunk_id"], "")
            self.assertEqual(child["next_chunk_id"], "")

    def test_context_header_is_not_persisted_to_content(self):
        conn = self.make_conn()
        diagnostics = Diagnostics(selected_tier="heading", tier_chain=["heading"])
        chunks = [Chunk(content="body", context_header="# Heading", seq=0, start=2, end=6)]

        with patch(
            "tiny_rag.persistence.chunk_persist_service.split_with_diagnostics",
            return_value=(chunks, diagnostics),
        ):
            result = persist_text_chunks(
                conn,
                tenant_id=TENANT_ID,
                knowledge_id=KNOWLEDGE_ID,
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                text="xxbody",
                config=SplitterConfig(strategy="heading"),
            )

        self.assertEqual(result.inserted_chunks[0].content, "body")
        self.assertEqual(result.inserted_chunks[0].context_header, "# Heading")
        self.assertEqual(result.inserted_chunks[0].row.content, "body")
        self.assertNotIn("# Heading", result.inserted_chunks[0].content)
        column_names = {
            row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        self.assertNotIn("context_header", column_names)
        rows = self.active_rows(conn)
        self.assertEqual(rows[0]["content"], "body")
        self.assertNotIn("# Heading", rows[0]["content"])

    def test_parent_child_context_header_is_retained_in_result_child(self):
        conn = self.make_conn()
        diagnostics = Diagnostics(selected_tier="heading", tier_chain=["heading"])
        split_result = ParentChildResult(
            parents=[
                Chunk(content="parent body", context_header="# Parent", seq=0, start=0, end=11)
            ],
            children=[
                ChildChunk(
                    content="child body",
                    context_header="# Parent\n## Child",
                    seq=1,
                    start=0,
                    end=10,
                    parent_index=0,
                )
            ],
        )

        with patch(
            "tiny_rag.persistence.chunk_persist_service.split_with_diagnostics",
            return_value=(split_result, diagnostics),
        ):
            result = persist_text_chunks(
                conn,
                tenant_id=TENANT_ID,
                knowledge_id=KNOWLEDGE_ID,
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                text="child body",
                config=SplitterConfig(parent_child=True, strategy="heading"),
            )

        text_children = [
            item for item in result.inserted_chunks if item.row.chunk_type == "text"
        ]
        self.assertEqual(len(text_children), 1)
        self.assertEqual(text_children[0].context_header, "# Parent\n## Child")
        self.assertEqual(text_children[0].row.content, "child body")

    def test_invalid_utf8_surrogates_and_nul_are_cleaned_before_insert(self):
        conn = self.make_conn()
        diagnostics = Diagnostics(selected_tier="legacy", tier_chain=["legacy"])
        chunks = [Chunk(content="a\x00b\ud800c", seq=0, start=0, end=5)]

        with patch(
            "tiny_rag.persistence.chunk_persist_service.split_with_diagnostics",
            return_value=(chunks, diagnostics),
        ):
            result = persist_text_chunks(
                conn,
                tenant_id=TENANT_ID,
                knowledge_id=KNOWLEDGE_ID,
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                text="a\x00bc",
            )

        rows = self.active_rows(conn)
        self.assertEqual(result.inserted_chunks[0].content, "abc")
        self.assertEqual(rows[0]["content"], "abc")

    def test_blank_chunks_are_not_persisted(self):
        conn = self.make_conn()
        diagnostics = Diagnostics(selected_tier="legacy", tier_chain=["legacy"])
        chunks = [
            Chunk(content="   ", seq=0, start=0, end=3),
            Chunk(content="real", seq=1, start=3, end=7),
        ]

        with patch(
            "tiny_rag.persistence.chunk_persist_service.split_with_diagnostics",
            return_value=(chunks, diagnostics),
        ):
            result = persist_text_chunks(
                conn,
                tenant_id=TENANT_ID,
                knowledge_id=KNOWLEDGE_ID,
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                text="   real",
            )

        rows = self.active_rows(conn)
        self.assertEqual(result.inserted_count, 1)
        self.assertEqual(result.stats.skipped_empty_count, 1)
        self.assertEqual([row["content"] for row in rows], ["real"])

    def test_repeat_persist_soft_deletes_old_active_chunks(self):
        conn = self.make_conn()

        first = persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=normal_text(),
            config=normal_config(),
        )
        first_ids = {chunk.id for chunk in first.inserted_chunks}

        second = persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=("new alpha beta gamma\n\n" * 3).strip() + "\n",
            config=normal_config(),
        )

        active = self.active_rows(conn)
        active_ids = {row["id"] for row in active}
        deleted_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE tenant_id = ?
              AND knowledge_id = ?
              AND deleted_at IS NOT NULL
            """,
            (TENANT_ID, KNOWLEDGE_ID),
        ).fetchone()[0]
        self.assertEqual(second.stats.deleted_count, first.inserted_count)
        self.assertEqual(len(active), second.inserted_count)
        self.assertTrue(first_ids.isdisjoint(active_ids))
        self.assertEqual(deleted_count, first.inserted_count)

    def test_seq_id_is_unique_and_increasing_across_soft_deletes(self):
        conn = self.make_conn()

        first = persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=normal_text(),
            config=normal_config(),
        )
        second = persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=("new alpha beta gamma\n\n" * 4).strip() + "\n",
            config=normal_config(),
        )

        seq_ids = [
            row["seq_id"]
            for row in conn.execute("SELECT seq_id FROM chunks ORDER BY seq_id ASC").fetchall()
        ]
        self.assertEqual(seq_ids, list(range(1, len(seq_ids) + 1)))
        self.assertGreater(second.inserted_chunks[0].seq_id, first.inserted_chunks[-1].seq_id)
        self.assertEqual(len(seq_ids), len(set(seq_ids)))

    def test_more_than_100_chunks_are_inserted_completely(self):
        conn = self.make_conn()
        diagnostics = Diagnostics(selected_tier="legacy", tier_chain=["legacy"])
        chunks = [
            Chunk(content=f"chunk-{index}", seq=index, start=index, end=index + 1)
            for index in range(105)
        ]

        with patch(
            "tiny_rag.persistence.chunk_persist_service.split_with_diagnostics",
            return_value=(chunks, diagnostics),
        ):
            result = persist_text_chunks(
                conn,
                tenant_id=TENANT_ID,
                knowledge_id=KNOWLEDGE_ID,
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                text="x" * 105,
            )

        rows = self.active_rows(conn)
        self.assertEqual(result.inserted_count, 105)
        self.assertEqual(len(rows), 105)
        self.assertEqual([row["seq_id"] for row in rows], list(range(1, 106)))

    def test_schema_contains_only_chunks_storage_tables(self):
        conn = self.make_conn()
        persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=normal_text(),
            config=normal_config(),
        )

        table_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        self.assertEqual(table_names, {"chunks"})

        persistence_dir = Path(__file__).resolve().parents[2] / "tiny_rag" / "persistence"
        source = "\n".join(path.read_text() for path in persistence_dir.glob("*.py"))
        self.assertNotIn("vector", source.lower())
        self.assertNotIn("embedding", source.lower())

    def test_repository_lists_active_chunks_only_by_default(self):
        conn = self.make_conn()
        persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=normal_text(),
            config=normal_config(),
        )
        persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=("replacement\n\n" * 3).strip() + "\n",
            config=normal_config(),
        )

        repo = ChunkRepository(conn)
        active = repo.list_chunks_by_knowledge_id(
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
        )
        all_rows = repo.list_chunks_by_knowledge_id(
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            include_deleted=True,
        )
        self.assertLess(len(active), len(all_rows))
        self.assertTrue(all(row.deleted_at is None for row in active))

    def test_repository_list_chunks_by_knowledge_id_returns_text_only(self):
        conn = self.make_conn()
        persist_text_chunks(
            conn,
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            text=("alpha beta gamma delta epsilon\n\n" * 20).strip() + "\n",
            config=SplitterConfig(
                parent_child=True,
                parent_chunk_size=1000,
                child_chunk_size=60,
                chunk_overlap=1,
                strategy="legacy",
            ),
        )

        repo = ChunkRepository(conn)
        text_chunks = repo.list_chunks_by_knowledge_id(
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
        )
        parent_chunks = repo.list_parent_chunks_by_knowledge_id(
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
        )
        all_chunks = repo.list_all_chunks_by_knowledge_id(
            tenant_id=TENANT_ID,
            knowledge_id=KNOWLEDGE_ID,
        )

        self.assertTrue(text_chunks)
        self.assertTrue(parent_chunks)
        self.assertTrue(all_chunks)
        self.assertTrue(all(row.chunk_type == "text" for row in text_chunks))
        self.assertTrue(all(row.chunk_type == "parent_text" for row in parent_chunks))
        self.assertEqual(len(all_chunks), len(text_chunks) + len(parent_chunks))
