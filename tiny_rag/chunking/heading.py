from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from .legacy import split_legacy
from .types import Chunk, DocumentProfile, SplitterConfig
from .utils import Heading, common_line_prefix, find_markdown_headings


@dataclass(frozen=True)
class HeadingBoundary:
    start: int
    line: str = ""


@dataclass(frozen=True)
class BreadcrumbPoint:
    offset: int
    breadcrumb: str


class HeadingHierarchy:
    def __init__(self, stack: Optional[List[Heading]] = None) -> None:
        self.stack: List[Heading] = list(stack or [])

    def copy(self) -> "HeadingHierarchy":
        return HeadingHierarchy(self.stack)

    def observe(self, heading: Heading) -> None:
        self.stack = [item for item in self.stack if item.level < heading.level]
        self.stack.append(heading)

    def breadcrumb_with_hashes(self) -> str:
        return "\n".join(heading.markdown for heading in self.stack)


def split_heading(
    text: str,
    cfg: SplitterConfig,
    profile: Optional[DocumentProfile] = None,
) -> List[Chunk]:
    cfg = cfg.normalized(default_overlap=False)
    if not text:
        return []

    headings = find_markdown_headings(text)
    primary_level = _primary_level(headings, profile)
    if primary_level == 0:
        return split_legacy(text, cfg)

    boundaries = _find_heading_boundaries(text, headings, primary_level)
    if len(boundaries) <= 1:
        return split_legacy(text, cfg)

    hierarchy = HeadingHierarchy()
    chunks: List[Chunk] = []
    seq = 0

    for index, boundary in enumerate(boundaries):
        end = boundaries[index + 1].start if index + 1 < len(boundaries) else len(text)
        if boundary.line:
            hierarchy.observe(_heading_from_boundary(boundary))

        breadcrumb = hierarchy.breadcrumb_with_hashes()
        section_start_hierarchy = hierarchy.copy()
        section = text[boundary.start : end]
        if not section:
            continue

        _observe_subheadings(section, primary_level, hierarchy)

        budget_len = len(breadcrumb) + (2 if breadcrumb else 0) + len(section)
        if budget_len <= cfg.chunk_size:
            chunks.append(
                Chunk(
                    content=section,
                    context_header=breadcrumb,
                    seq=seq,
                    start=boundary.start,
                    end=end,
                )
            )
            seq += 1
            continue

        sub_breadcrumbs = _section_breadcrumbs(section, primary_level, section_start_hierarchy)
        for sub_chunk in split_legacy(section, cfg):
            chunks.append(
                Chunk(
                    content=sub_chunk.content,
                    context_header=_breadcrumb_at_offset(
                        sub_breadcrumbs,
                        sub_chunk.start,
                        breadcrumb,
                    ),
                    seq=seq,
                    start=boundary.start + sub_chunk.start,
                    end=boundary.start + sub_chunk.end,
                )
            )
            seq += 1

    return _assign_sequence(_coalesce_tiny_chunks(chunks, cfg))


def _primary_level(headings: List[Heading], profile: Optional[DocumentProfile]) -> int:
    if profile and profile.primary_heading_level:
        return profile.primary_heading_level
    if not headings:
        return 0
    return _choose_primary_level(headings)


def _choose_primary_level(headings: List[Heading]) -> int:
    counts = Counter(heading.level for heading in headings)
    frequent = [level for level, count in counts.items() if count >= 3]
    if frequent:
        return min(frequent)
    return max(counts)


def _find_heading_boundaries(
    text: str,
    headings: List[Heading],
    primary_level: int,
) -> List[HeadingBoundary]:
    by_start = {
        heading.start: heading for heading in headings if heading.level <= primary_level
    }
    boundaries: List[HeadingBoundary] = [
        HeadingBoundary(start=0, line=by_start[0].markdown if 0 in by_start else "")
    ]
    boundaries.extend(
        HeadingBoundary(start=heading.start, line=heading.markdown)
        for heading in headings
        if heading.level <= primary_level and heading.start != 0
    )
    boundaries = sorted(boundaries, key=lambda boundary: boundary.start)
    return _dedupe_boundaries(boundaries)


def _dedupe_boundaries(boundaries: List[HeadingBoundary]) -> List[HeadingBoundary]:
    if not boundaries:
        return []
    deduped: List[HeadingBoundary] = []
    for boundary in boundaries:
        if deduped and deduped[-1].start == boundary.start:
            if boundary.line:
                deduped[-1] = boundary
        else:
            deduped.append(boundary)
    return deduped


def _heading_from_boundary(boundary: HeadingBoundary) -> Heading:
    hashes, _, title = boundary.line.partition(" ")
    return Heading(level=len(hashes), title=title.strip(), start=boundary.start, end=boundary.start)


def _observe_subheadings(
    section: str,
    primary_level: int,
    hierarchy: HeadingHierarchy,
) -> None:
    for heading in find_markdown_headings(section):
        if heading.start == 0:
            continue
        if heading.level > primary_level:
            hierarchy.observe(heading)


def _section_breadcrumbs(
    section: str,
    primary_level: int,
    initial_hierarchy: HeadingHierarchy,
) -> List[BreadcrumbPoint]:
    hierarchy = initial_hierarchy.copy()
    points = [BreadcrumbPoint(offset=0, breadcrumb=hierarchy.breadcrumb_with_hashes())]
    for heading in find_markdown_headings(section):
        if heading.start == 0:
            continue
        if heading.level > primary_level:
            hierarchy.observe(heading)
            points.append(
                BreadcrumbPoint(
                    offset=heading.start,
                    breadcrumb=hierarchy.breadcrumb_with_hashes(),
                )
            )
    return points


def _breadcrumb_at_offset(
    points: List[BreadcrumbPoint],
    offset: int,
    default: str,
) -> str:
    active = default
    for point in points:
        if point.offset > offset:
            break
        active = point.breadcrumb
    return active


def _coalesce_tiny_chunks(chunks: List[Chunk], cfg: SplitterConfig) -> List[Chunk]:
    if not chunks:
        return []
    target = max(cfg.chunk_size // 2, 200)
    merged: List[Chunk] = [chunks[0]]
    for next_chunk in chunks[1:]:
        current = merged[-1]
        prefix = common_line_prefix(current.context_header, next_chunk.context_header)
        current_len = len(current.content)
        next_len = len(next_chunk.content)
        if (
            prefix
            and current.end == next_chunk.start
            and current_len < target
            and current_len + next_len <= cfg.chunk_size
        ):
            current.content += next_chunk.content
            current.context_header = "\n".join(prefix)
            current.end = next_chunk.end
        else:
            merged.append(next_chunk)
    return merged


def _assign_sequence(chunks: List[Chunk]) -> List[Chunk]:
    for index, chunk in enumerate(chunks):
        chunk.seq = index
    return chunks
