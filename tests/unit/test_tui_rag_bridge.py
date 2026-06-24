from __future__ import annotations

import io
import json
import sqlite3
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

import ingest as offline_ingest
from scripts import tui_rag_bridge


ROOT = Path(__file__).resolve().parents[2]


class TuiRagBridgeTests(TestCase):
    def test_offline_ingest_then_online_bridge_reads_existing_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tui-rag.sqlite")
            source = ROOT / "tests/fixtures/documents/markdown/sample_manual.md"
            fixture = Path(tmp) / "sample_manual.md"
            fixture.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

            init_payload = offline_ingest.ingest_documents(
                db=db_path,
                documents=[fixture],
            )

            self.assertTrue(init_payload["success"])
            self.assertEqual(init_payload["knowledge_base"]["id"], "local-tui-kb")
            self.assertEqual(
                sorted(tool["function"]["name"] for tool in init_payload["tools"]),
                ["grep_chunks", "knowledge_search"],
            )

            fixture.unlink()
            definitions_payload = run_bridge(["definitions", "--db", db_path])

            self.assertTrue(definitions_payload["success"])
            self.assertEqual(
                sorted(tool["function"]["name"] for tool in definitions_payload["tools"]),
                ["grep_chunks", "knowledge_search"],
            )
            self.assertEqual(definitions_payload["knowledge_base"]["id"], "local-tui-kb")
            self.assertIn("sample_manual", definitions_payload["selected_documents"][0]["title"])

            grep_payload = run_bridge(
                ["execute", "--db", db_path, "--tool", "grep_chunks"],
                stdin={"query": "chunk_size"},
            )

            self.assertTrue(grep_payload["success"])
            self.assertEqual(grep_payload["data"]["display_type"], "grep_results")
            self.assertGreaterEqual(grep_payload["data"]["result_count"], 1)
            self.assertIn("<grep_results", grep_payload["output"])
            self.assertIn("chunk_size", grep_payload["output"])

            search_payload = run_bridge(
                ["execute", "--db", db_path, "--tool", "knowledge_search"],
                stdin={"queries": ["What options does the manual describe for chunking?"]},
            )

            self.assertTrue(search_payload["success"])
            self.assertIn("sample_manual", init_payload["selected_documents"][0]["title"])
            self.assertIn("chunk_size", search_payload["output"])

    def test_ingest_script_discovers_docs_dir_and_writes_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "knowledge"
            docs_dir.mkdir()
            (docs_dir / "manual.md").write_text("# Manual\n\nchunk_size is configurable.", encoding="utf-8")
            db_path = str(Path(tmp) / "tui-rag.sqlite")

            out = io.StringIO()
            with redirect_stdout(out):
                status = offline_ingest.main(
                    [
                        "--docs-dir",
                        str(docs_dir),
                        "--db",
                        db_path,
                        "--json",
                    ]
                )

            payload = json.loads(out.getvalue())
            self.assertEqual(status, 0)
            self.assertTrue(payload["success"])
            self.assertTrue(Path(db_path).exists())
            self.assertTrue(Path(db_path + ".manifest.json").exists())
            self.assertEqual(payload["selected_documents"][0]["file_name"], "manual.md")

    def test_ingest_script_allow_empty_writes_watchable_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "knowledge"
            db_path = str(Path(tmp) / "tui-rag.sqlite")

            out = io.StringIO()
            with redirect_stdout(out):
                status = offline_ingest.main(
                    [
                        "--docs-dir",
                        str(docs_dir),
                        "--db",
                        db_path,
                        "--allow-empty",
                        "--json",
                    ]
                )

            payload = json.loads(out.getvalue())
            self.assertEqual(status, 0)
            self.assertTrue(payload["success"])
            self.assertEqual(payload["selected_documents"], [])
            self.assertTrue(docs_dir.exists())
            self.assertTrue(Path(db_path).exists())
            self.assertTrue(Path(db_path + ".manifest.json").exists())

            definitions_payload = run_bridge(["definitions", "--db", db_path])
            self.assertTrue(definitions_payload["success"])
            self.assertEqual(definitions_payload["selected_documents"], [])
            self.assertEqual(
                sorted(tool["function"]["name"] for tool in definitions_payload["tools"]),
                ["grep_chunks", "knowledge_search"],
            )

    def test_offline_ingest_syncs_changed_and_deleted_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "knowledge"
            docs_dir.mkdir()
            doc_a = docs_dir / "a.md"
            doc_a.write_text("# A\n\nalpha one.", encoding="utf-8")
            db_path = str(Path(tmp) / "tui-rag.sqlite")

            first = offline_ingest.ingest_documents(db=db_path, documents=[doc_a])
            first_id = first["selected_documents"][0]["knowledge_id"]

            doc_a.write_text("# A\n\ngamma two.", encoding="utf-8")
            second = offline_ingest.ingest_documents(db=db_path, documents=[doc_a])
            self.assertEqual(second["selected_documents"][0]["knowledge_id"], first_id)

            doc_a.unlink()
            doc_b = docs_dir / "b.md"
            doc_b.write_text("# B\n\nbeta three.", encoding="utf-8")
            third = offline_ingest.ingest_documents(db=db_path, documents=[doc_b])

            self.assertEqual(len(third["selected_documents"]), 1)
            self.assertEqual(third["selected_documents"][0]["file_name"], "b.md")
            self.assertNotEqual(third["selected_documents"][0]["knowledge_id"], first_id)

            conn = sqlite3.connect(db_path)
            try:
                active = conn.execute(
                    "SELECT file_name FROM knowledges WHERE deleted_at IS NULL"
                ).fetchall()
                deleted = conn.execute(
                    "SELECT file_name FROM knowledges WHERE deleted_at IS NOT NULL"
                ).fetchall()
            finally:
                conn.close()

            self.assertEqual(active, [("b.md",)])
            self.assertEqual(deleted, [("a.md",)])


def run_bridge(args: list[str], *, stdin: dict | None = None) -> dict:
    old_stdin = sys.stdin
    out = io.StringIO()
    try:
        sys.stdin = io.StringIO(json.dumps(stdin or {}))
        with redirect_stdout(out):
            status = tui_rag_bridge.main(args)
    finally:
        sys.stdin = old_stdin

    payload = json.loads(out.getvalue())
    if status != 0:
        raise AssertionError(payload.get("error") or f"bridge exited with {status}")
    return payload
