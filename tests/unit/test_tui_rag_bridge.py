from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

from scripts import tui_rag_bridge


ROOT = Path(__file__).resolve().parents[2]


class TuiRagBridgeTests(TestCase):
    def test_init_registers_tools_and_execute_reads_indexed_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tui-rag.sqlite")
            fixture = str(ROOT / "tests/fixtures/documents/markdown/sample_manual.md")

            init_payload = run_bridge(
                [
                    "init",
                    "--db",
                    db_path,
                    "--document",
                    fixture,
                ]
            )

            self.assertTrue(init_payload["success"])
            self.assertEqual(init_payload["knowledge_base"]["id"], "local-tui-kb")
            self.assertEqual(
                sorted(tool["function"]["name"] for tool in init_payload["tools"]),
                ["grep_chunks", "knowledge_search"],
            )

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
