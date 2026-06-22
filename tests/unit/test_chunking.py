import re
import subprocess
import sys
import unittest
from pathlib import Path

from tiny_rag.chunking import (
    Chunk,
    ParentChildResult,
    SplitterConfig,
    split,
    split_parent_child,
    split_with_diagnostics,
)
from tiny_rag.chunking.heading import split_heading
from tiny_rag.chunking.heuristic import _find_boundaries, split_heuristic
from tiny_rag.chunking.legacy import HeaderTracker, HeaderTrackerHook, split_legacy
from tiny_rag.chunking.patterns import NUMBERED_HEADING_RE, PAGE_FOOTER_RE, chapter_marker_kind
from tiny_rag.chunking.profile import profile_document
from tiny_rag.chunking.strategy import _merge_breadcrumbs, validate_chunks


class ChunkingTests(unittest.TestCase):
    def assert_offsets_match(self, text, chunks):
        for chunk in chunks:
            self.assertEqual(text[chunk.start : chunk.end], chunk.content)
            self.assertEqual(chunk.end - chunk.start, len(chunk.content))

    def test_legacy_preserves_rune_offsets_for_mixed_text(self):
        text = "第一段🙂内容。\n\nSecond paragraph with ascii words.\n\n第三段结尾。"
        chunks = split_legacy(text, SplitterConfig(chunk_size=24, chunk_overlap=0))
        self.assertGreater(len(chunks), 1)
        self.assert_offsets_match(text, chunks)

    def test_public_config_defaults_non_positive_overlap(self):
        zero_cfg = SplitterConfig(chunk_size=200, chunk_overlap=0).normalized()
        negative_cfg = SplitterConfig(chunk_size=200, chunk_overlap=-1).normalized()
        self.assertEqual(zero_cfg.chunk_overlap, 80)
        self.assertEqual(negative_cfg.chunk_overlap, 80)

    def test_direct_splitter_config_preserves_zero_overlap(self):
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=0).normalized(default_overlap=False)
        self.assertEqual(cfg.chunk_overlap, 0)

    def test_overlap_is_capped_to_half_chunk_size(self):
        cfg = SplitterConfig(chunk_size=101, chunk_overlap=100).normalized()
        self.assertEqual(cfg.chunk_overlap, 50)

    def test_parent_child_overlap_uses_child_size_for_config_cap(self):
        cfg = SplitterConfig(
            chunk_size=100,
            chunk_overlap=800,
            parent_child=True,
            child_chunk_size=1000,
        ).normalized()
        self.assertEqual(cfg.chunk_overlap, 500)

    def test_embedding_content_trims_body(self):
        self.assertEqual(Chunk(" \n body \n ").embedding_content(), "body")
        self.assertEqual(
            Chunk(" \n body \n ", context_header="# Heading").embedding_content(),
            "# Heading\n\nbody",
        )

    def test_profile_heading_density_uses_line_count(self):
        text = "# A\nbody\n## B\nbody\n## C\nbody\n"
        profile = profile_document(text)
        self.assertEqual(profile.heading_count, 3)
        self.assertEqual(profile.total_lines, 7)
        self.assertAlmostEqual(profile.heading_density, 3 / 7)

    def test_legacy_keeps_code_block_together_when_possible(self):
        text = "intro\n\n```python\n# not a heading\nprint('hello')\n```\n\noutro"
        chunks = split_legacy(text, SplitterConfig(chunk_size=80, chunk_overlap=0))
        joined = "\n---\n".join(chunk.content for chunk in chunks)
        self.assertIn("```python\n# not a heading\nprint('hello')\n```", joined)
        self.assert_offsets_match(text, chunks)

    def test_direct_legacy_zero_overlap_has_no_range_overlap(self):
        text = "alpha beta gamma\n\n" * 8
        chunks = split_legacy(text, SplitterConfig(chunk_size=35, chunk_overlap=0))
        self.assertGreater(len(chunks), 1)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertLessEqual(previous.end, current.start)

    def test_heading_context_and_code_fence_heading_ignore(self):
        text = (
            "# Manual\n"
            "intro\n\n"
            "## Install\n"
            "steps\n\n"
            "```md\n"
            "## Not Heading\n"
            "```\n\n"
            "## Configure\n"
            "settings\n"
        )
        chunks = split_heading(text, SplitterConfig(chunk_size=50, chunk_overlap=0))
        self.assertTrue(any("## Install" in chunk.context_header for chunk in chunks))
        self.assertFalse(any("## Not Heading" in chunk.context_header for chunk in chunks))
        self.assert_offsets_match(text, chunks)

    def test_heading_short_section_uses_section_boundary_breadcrumb(self):
        text = (
            "# Manual\n"
            + "overview " * 7
            + "\n"
            "## Install\n"
            "intro\n"
            "### Docker\n"
            + "short details " * 4
            + "\n"
            "## Configure\n"
            "settings\n"
            "## Run\n"
            "done\n"
        )
        chunks = split_heading(text, SplitterConfig(chunk_size=106, chunk_overlap=0))
        install_chunk = next(chunk for chunk in chunks if "### Docker" in chunk.content)
        self.assertEqual(install_chunk.context_header, "# Manual\n## Install")

    def test_heading_long_section_subchunks_track_deep_subheadings(self):
        text = (
            "# Manual\n"
            "## Install\n"
            + "intro sentence. " * 12
            + "\n### Docker\n"
            + "docker steps. " * 20
            + "\n## Configure\n"
            + "settings\n"
            + "## Run\n"
            + "done\n"
        )
        chunks = split_heading(
            text,
            SplitterConfig(
                chunk_size=120,
                chunk_overlap=0,
                separators=["\n\n", "\n", ". "],
            ),
        )
        docker_chunks = [chunk for chunk in chunks if "docker steps" in chunk.content]
        self.assertTrue(docker_chunks)
        self.assertTrue(
            any(chunk.context_header == "# Manual\n## Install\n### Docker" for chunk in docker_chunks)
        )

    def test_auto_routes_markdown_to_heading(self):
        text = (
            "# Guide\nintro\n"
            "## A\nalpha\n"
            "## B\nbeta\n"
            "## C\ngamma\n"
            "## D\ndelta\n"
        )
        chunks, diag = split_with_diagnostics(
            text,
            SplitterConfig(chunk_size=40, chunk_overlap=0, strategy="auto"),
        )
        self.assertIn("heading", diag.tier_chain)
        self.assertEqual(diag.selected_tier, "heading")
        self.assert_offsets_match(text, chunks)

    def test_empty_strategy_uses_legacy_without_profile(self):
        text = (
            "# Guide\nintro\n"
            "## A\nalpha\n"
            "## B\nbeta\n"
            "## C\ngamma\n"
        )
        chunks, diag = split_with_diagnostics(text, SplitterConfig(chunk_size=40, strategy=""))
        self.assertEqual(diag.tier_chain, ["legacy"])
        self.assertEqual(diag.selected_tier, "legacy")
        self.assertIsNone(diag.profile)
        self.assert_offsets_match(text, chunks)

    def test_unknown_strategy_uses_auto_router(self):
        text = (
            "# Guide\nintro\n"
            "## A\nalpha\n"
            "## B\nbeta\n"
            "## C\ngamma\n"
            "## D\ndelta\n"
        )
        chunks, diag = split_with_diagnostics(
            text,
            SplitterConfig(chunk_size=40, strategy="mystery"),
        )
        self.assertIn("heading", diag.tier_chain)
        self.assertEqual(diag.selected_tier, "heading")
        self.assertIsNotNone(diag.profile)
        self.assert_offsets_match(text, chunks)

    def test_heuristic_uses_pdf_like_boundaries(self):
        text = (
            "Chapter 1 Overview\n"
            + "a" * 60
            + "\n\f"
            + "Page 1 of 2\n"
            + "Chapter 2 Details\n"
            + "b" * 60
        )
        chunks = split_heuristic(text, SplitterConfig(chunk_size=90, chunk_overlap=10))
        self.assertGreaterEqual(len(chunks), 2)
        self.assert_offsets_match(text, chunks)

    def test_heuristic_overlap_boundary_progresses(self):
        script = """
from tiny_rag.chunking import SplitterConfig
from tiny_rag.chunking.heuristic import split_heuristic

text = "a" * 50 + "\\f" + "b" * 60
chunks = split_heuristic(text, SplitterConfig(chunk_size=100, chunk_overlap=50))
assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 50), (50, 111)]
for chunk in chunks:
    assert text[chunk.start:chunk.end] == chunk.content
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                timeout=2,
            )
        except subprocess.TimeoutExpired:
            self.fail("heuristic splitter did not make progress with overlap-aligned boundary")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_form_feed_boundary_is_at_symbol_position(self):
        text = "before\fafter"
        form_feed_pos = text.index("\f")
        form_feed_boundaries = [
            boundary.pos
            for boundary in _find_boundaries(text)
            if boundary.kind == "form_feed"
        ]
        self.assertEqual(form_feed_boundaries, [form_feed_pos])

    def test_numbered_heading_pattern_accepts_wider_titles(self):
        accepted = [
            "1.1 中文标题",
            "2.3.4 lowercase title",
            "IV. 结果",
            "12. Anything",
        ]
        rejected = [
            "1 single-level without dot",
            "1.2.3.4.5 too deep",
        ]
        for line in accepted:
            self.assertIsNotNone(NUMBERED_HEADING_RE.match(line), line)
        for line in rejected:
            self.assertIsNone(NUMBERED_HEADING_RE.match(line), line)

    def test_page_footer_pattern_accepts_english_german_chinese(self):
        accepted = [
            "Page 3 of 10",
            "Page 3/10",
            "Seite 4 von 12",
            "页码 2/9",
            "页 8",
        ]
        for line in accepted:
            self.assertIsNotNone(PAGE_FOOTER_RE.match(line), line)

    def test_chapter_marker_respects_language_filter(self):
        self.assertEqual(chapter_marker_kind("Chapter IV. Intro", ["en"]), "en")
        self.assertEqual(chapter_marker_kind("Abschnitt 2. Überblick", ["de"]), "de")
        self.assertEqual(chapter_marker_kind("第 3 节 方法", ["zh"]), "zh")
        self.assertEqual(chapter_marker_kind("Kapitel 1. Überblick", ["en"]), "")
        self.assertEqual(chapter_marker_kind("Section 1. Intro", ["de"]), "")

    def test_heuristic_chapter_boundaries_respect_language_filter(self):
        text = "Kapitel 1. Überblick\nalpha\nSection 2. Details\nbeta\n"
        de_boundaries = [
            boundary.pos
            for boundary in _find_boundaries(text, languages=["de"])
            if boundary.kind == "chapter_marker"
        ]
        en_boundaries = [
            boundary.pos
            for boundary in _find_boundaries(text, languages=["en"])
            if boundary.kind == "chapter_marker"
        ]
        self.assertEqual(de_boundaries, [0])
        self.assertEqual(en_boundaries, [text.index("Section")])

    def test_validate_rejects_single_large_chunk(self):
        text = "x" * 200
        ok, reason = validate_chunks(
            [split(text, SplitterConfig(chunk_size=1000, strategy="legacy"))[0]],
            len(text),
            SplitterConfig(chunk_size=50),
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "single chunk for large document")

    def test_parent_child_offsets_and_parent_index(self):
        text = "# Doc\n\n## One\n" + "alpha " * 40 + "\n\n## Two\n" + "beta " * 40
        result = split_parent_child(
            text,
            SplitterConfig(chunk_size=120, chunk_overlap=0, strategy="heading"),
            SplitterConfig(chunk_size=45, chunk_overlap=5, strategy="legacy"),
        )
        self.assertGreater(len(result.children), 2)
        for child in result.children:
            if child.parent_index != -1:
                self.assertGreaterEqual(child.parent_index, 0)
                self.assertLess(child.parent_index, len(result.parents))
            self.assertEqual(text[child.start : child.end], child.content)

    def test_split_uses_parent_child_config_flag(self):
        text = ("alpha beta gamma\n\n" * 20).strip() + "\n"
        cfg = SplitterConfig(
            parent_child=True,
            parent_chunk_size=1000,
            child_chunk_size=60,
            chunk_overlap=0,
            strategy="legacy",
        )

        result, diagnostics = split_with_diagnostics(text, cfg)

        self.assertIsInstance(split(text, cfg), ParentChildResult)
        self.assertIsInstance(result, ParentChildResult)
        self.assertEqual(diagnostics.selected_tier, "parent_child")
        self.assertEqual(diagnostics.tier_chain, ["parent_child"])
        self.assertEqual(len(result.parents), 1)
        self.assertGreater(len(result.children), 1)
        for child in result.children:
            self.assertEqual(text[child.start : child.end], child.content)

    def test_parent_child_omits_unsplit_parent(self):
        text = "# Doc\n\nShort body.\n"
        result = split_parent_child(
            text,
            SplitterConfig(chunk_size=1000, chunk_overlap=0, strategy="legacy"),
            SplitterConfig(chunk_size=1000, chunk_overlap=0, strategy="legacy"),
        )
        self.assertEqual(result.parents, [])
        self.assertEqual(len(result.children), 1)
        self.assertEqual(result.children[0].parent_index, -1)
        self.assertEqual(result.children[0].content, text)

    def test_parent_child_keeps_parent_only_when_child_split_changes_it(self):
        text = ("alpha beta gamma\n\n" * 20).strip() + "\n"
        result = split_parent_child(
            text,
            SplitterConfig(chunk_size=1000, chunk_overlap=0, strategy="legacy"),
            SplitterConfig(chunk_size=60, chunk_overlap=0, strategy="legacy"),
        )
        self.assertEqual(len(result.parents), 1)
        self.assertGreater(len(result.children), 1)
        self.assertTrue(all(child.parent_index == 0 for child in result.children))

    def test_parent_child_table_context_preserves_child_offsets(self):
        header = "| Name | Value |\n|---|---|\n"
        rows = "".join(f"| row{i} | value{i} |\n" for i in range(8))
        text = header + rows
        result = split_parent_child(
            text,
            SplitterConfig(chunk_size=70, chunk_overlap=0, separators=["\n"], strategy="legacy"),
            SplitterConfig(chunk_size=40, chunk_overlap=0, separators=["\n"], strategy="legacy"),
        )
        self.assertTrue(result.children)
        for child in result.children:
            self.assertEqual(text[child.start : child.end], child.content)
        row_children = [child for child in result.children if "| row" in child.content]
        self.assertTrue(row_children)
        self.assertTrue(all(child.context_header == header.strip() for child in row_children))

    def test_merge_breadcrumbs_dedupes_parent_tail_against_child_head(self):
        merged = _merge_breadcrumbs("# Doc\n## Install", "  ## Install  \n### Docker")
        self.assertEqual(merged, "# Doc\n## Install\n### Docker")
        self.assertEqual(_merge_breadcrumbs("# Doc\n## Install", "## Install"), "# Doc\n## Install")

    def test_legacy_prepends_table_header_to_later_table_chunks(self):
        header = "| Name | Value |\n|---|---|\n"
        rows = "".join(f"| row{i} | value{i} |\n" for i in range(8))
        text = header + rows
        chunks = split_legacy(
            text,
            SplitterConfig(chunk_size=70, chunk_overlap=0, separators=["\n"]),
        )
        self.assertGreater(len(chunks), 2)
        later_table_chunks = [chunk for chunk in chunks[1:] if "| row" in chunk.content]
        self.assertTrue(later_table_chunks)
        self.assertTrue(all(chunk.context_header == header.strip() for chunk in later_table_chunks))
        self.assert_offsets_match(text, chunks)

    def test_legacy_extends_empty_table_header_from_first_data_row(self):
        empty_header = "||\n|---|---|\n"
        inferred_header = "| Column A | Column B |\n|---|---|\n"
        rows = "| Column A | Column B |\n" + "".join(f"| a{i} | b{i} |\n" for i in range(6))
        text = empty_header + rows
        chunks = split_legacy(
            text,
            SplitterConfig(chunk_size=62, chunk_overlap=0, separators=["\n"]),
        )
        later_chunks = [chunk for chunk in chunks[1:] if "| a" in chunk.content]
        self.assertTrue(later_chunks)
        self.assertTrue(all(chunk.context_header == inferred_header.strip() for chunk in later_chunks))
        self.assert_offsets_match(text, chunks)

    def test_legacy_table_paragraph_break_starts_fresh_header(self):
        first_header = "| A | B |\n|---|---|\n"
        second_header = "| C | D |\n|---|---|\n"
        text = (
            first_header
            + "| a1 | b1 |\n"
            + "\n\n"
            + second_header
            + "".join(f"| c{i} | d{i} |\n" for i in range(4))
        )
        chunks = split_legacy(
            text,
            SplitterConfig(chunk_size=48, chunk_overlap=0, separators=["\n\n", "\n"]),
        )
        second_table_chunks = [chunk for chunk in chunks if "| c" in chunk.content]
        self.assertTrue(second_table_chunks)
        for chunk in second_table_chunks:
            self.assertEqual(text[chunk.start : chunk.end], chunk.content)
            if not chunk.content.startswith(second_header):
                self.assertEqual(chunk.context_header, second_header.strip())
            self.assertNotEqual(chunk.context_header, first_header.strip())

    def test_legacy_table_column_mismatch_starts_new_header(self):
        first_header = "| A | B |\n|---|---|\n"
        second_header = "| C | D | E |\n|---|---|---|\n"
        text = (
            first_header
            + "| a1 | b1 |\n"
            + second_header
            + "".join(f"| c{i} | d{i} | e{i} |\n" for i in range(4))
        )
        chunks = split_legacy(
            text,
            SplitterConfig(chunk_size=70, chunk_overlap=0, separators=["\n"]),
        )
        second_table_chunks = [chunk for chunk in chunks if "| c" in chunk.content]
        self.assertTrue(second_table_chunks)
        for chunk in second_table_chunks:
            self.assertEqual(text[chunk.start : chunk.end], chunk.content)
            if not chunk.content.startswith(second_header):
                self.assertEqual(chunk.context_header, second_header.strip())
            self.assertNotEqual(chunk.context_header, first_header.strip())

    def test_header_tracker_returns_headers_by_priority(self):
        tracker = HeaderTracker()
        tracker.hooks = [
            HeaderTrackerHook(re.compile(r"^low$"), re.compile(r"^$"), 5),
            HeaderTrackerHook(re.compile(r"^high$"), re.compile(r"^$"), 20),
        ]
        tracker.update("low")
        tracker.update("high")
        self.assertEqual(tracker.get_headers(), "high\nlow")


if __name__ == "__main__":
    unittest.main()
