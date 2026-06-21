from __future__ import annotations

from typing import List

from .patterns import (
    BLANK_BLOCK_RE,
    NUMBERED_HEADING_RE,
    PAGE_FOOTER_RE,
    ROMAN_HEADING_RE,
    VISUAL_SEPARATOR_RE,
    chapter_marker_kind,
    is_all_caps_heading,
    is_table_row,
)
from .types import DocumentProfile
from .utils import choose_primary_heading_level, find_markdown_headings, is_fence_line, iter_lines_with_offsets


def profile_document(text: str, languages: List[str] = None) -> DocumentProfile:
    languages = list(languages or [])
    headings = find_markdown_headings(text)
    total_lines = text.count("\n") + 1 if text else 0
    profile = DocumentProfile(
        total_chars=len(text),
        total_lines=total_lines,
        heading_count=len(headings),
        heading_density=(len(headings) / total_lines) if total_lines else 0.0,
        primary_heading_level=choose_primary_heading_level(headings) if headings else None,
        form_feed_count=text.count("\f"),
        blank_block_count=len(BLANK_BLOCK_RE.findall(text)),
        languages=languages,
    )

    in_fence = False
    table_rows = 0
    for line, _, _ in iter_lines_with_offsets(text):
        if is_fence_line(line):
            profile.code_fence_count += 1
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if NUMBERED_HEADING_RE.match(stripped):
            profile.numbered_heading_count += 1
        elif ROMAN_HEADING_RE.match(stripped):
            profile.roman_heading_count += 1
        if chapter_marker_kind(stripped, languages):
            profile.chapter_marker_count += 1
        if is_all_caps_heading(stripped):
            profile.all_caps_heading_count += 1
        if VISUAL_SEPARATOR_RE.match(stripped):
            profile.visual_separator_count += 1
        if PAGE_FOOTER_RE.match(stripped):
            profile.page_footer_count += 1
        if is_table_row(stripped):
            table_rows += 1

    profile.table_count = table_rows
    return profile


def select_strategy(profile: DocumentProfile) -> List[str]:
    tiers: List[str] = []
    if (
        profile.heading_count >= 3
        and profile.heading_density > 0.005
        and profile.primary_heading_level is not None
    ):
        tiers.append("heading")

    if (
        profile.structure_marker_count >= 5
        or profile.form_feed_count > 0
        or profile.chapter_marker_count > 0
    ):
        tiers.append("heuristic")

    tiers.append("legacy")
    return _dedupe_preserving_order(tiers)


def _dedupe_preserving_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
