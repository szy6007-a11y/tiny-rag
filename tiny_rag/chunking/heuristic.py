from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .legacy import split_legacy
from .patterns import (
    BLANK_BLOCK_RE,
    NUMBERED_HEADING_RE,
    PAGE_FOOTER_RE,
    ROMAN_HEADING_RE,
    VISUAL_SEPARATOR_RE,
    chapter_marker_kind,
    is_all_caps_heading,
)
from .protected import is_inside_spans, protected_spans
from .types import Chunk, SplitterConfig
from .utils import is_fence_line, iter_lines_with_offsets


PRIORITY = {
    "blank_block": 1,
    "page_footer": 2,
    "visual_separator": 3,
    "all_caps_heading": 4,
    "chapter_marker": 5,
    "numbered_heading": 6,
    "form_feed": 7,
}


@dataclass(frozen=True)
class Boundary:
    pos: int
    kind: str

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.kind, 0)


def split_heuristic(text: str, cfg: SplitterConfig) -> List[Chunk]:
    cfg = cfg.normalized(default_overlap=False)
    if not text:
        return []

    boundaries = _find_boundaries(text, cfg.languages)
    if len(boundaries) <= 2 and len(text) > cfg.chunk_size:
        return split_legacy(text, cfg)

    chunks = _greedy_chunks(text, boundaries, cfg)
    return chunks


def _find_boundaries(text: str, languages=None) -> List[Boundary]:
    candidates: List[Boundary] = [Boundary(0, "start"), Boundary(len(text), "end")]
    candidates.extend(Boundary(index, "form_feed") for index, char in enumerate(text) if char == "\f")
    candidates.extend(Boundary(match.end(), "blank_block") for match in BLANK_BLOCK_RE.finditer(text))

    in_fence = False
    for line, start, _ in iter_lines_with_offsets(text):
        if is_fence_line(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if NUMBERED_HEADING_RE.match(stripped) or ROMAN_HEADING_RE.match(stripped):
            candidates.append(Boundary(start, "numbered_heading"))
        elif chapter_marker_kind(stripped, languages):
            candidates.append(Boundary(start, "chapter_marker"))
        elif is_all_caps_heading(stripped):
            candidates.append(Boundary(start, "all_caps_heading"))
        elif VISUAL_SEPARATOR_RE.match(stripped):
            candidates.append(Boundary(start, "visual_separator"))
        elif PAGE_FOOTER_RE.match(stripped):
            candidates.append(Boundary(start, "page_footer"))

    spans = protected_spans(text)
    candidates = [
        boundary
        for boundary in candidates
        if boundary.pos in {0, len(text)} or not is_inside_spans(boundary.pos, spans)
    ]
    return _dedupe_boundaries(candidates)


def _dedupe_boundaries(boundaries: List[Boundary]) -> List[Boundary]:
    by_pos: Dict[int, Boundary] = {}
    for boundary in boundaries:
        current = by_pos.get(boundary.pos)
        if current is None or boundary.priority > current.priority:
            by_pos[boundary.pos] = boundary
    return [by_pos[pos] for pos in sorted(by_pos)]


def _greedy_chunks(text: str, boundaries: List[Boundary], cfg: SplitterConfig) -> List[Chunk]:
    chunks: List[Chunk] = []
    min_chunk_size = max(cfg.chunk_size // 4, 50)
    positions = [boundary.pos for boundary in boundaries]
    boundary_positions = set(positions)

    chunk_start = positions[0]
    cur_end = chunk_start
    index = 1
    while index < len(positions):
        next_end = positions[index]
        if next_end <= cur_end:
            index += 1
            continue

        block_len = next_end - cur_end
        if block_len > cfg.chunk_size:
            if cur_end > chunk_start:
                _append_chunk(text, chunk_start, cur_end, chunks)
            chunks.extend(_legacy_subchunks(text, cur_end, next_end, cfg))
            chunk_start = next_end
            cur_end = next_end
            index += 1
            continue

        accumulated = next_end - chunk_start
        if accumulated <= cfg.chunk_size or cur_end == chunk_start:
            cur_end = next_end
            index += 1
            continue

        if cur_end - chunk_start >= min_chunk_size:
            _append_chunk(text, chunk_start, cur_end, chunks)
            chunk_start = _apply_overlap_aligned(text, cur_end, boundary_positions, cfg)
            if chunk_start >= cur_end:
                chunk_start = cur_end
            continue

        cur_end = next_end
        index += 1

    if cur_end > chunk_start:
        _append_chunk(text, chunk_start, cur_end, chunks)
    return chunks


def _append_chunk(text: str, start: int, end: int, chunks: List[Chunk]) -> None:
    if start < end:
        chunks.append(Chunk(content=text[start:end], start=start, end=end))


def _legacy_subchunks(text: str, start: int, end: int, cfg: SplitterConfig) -> List[Chunk]:
    result: List[Chunk] = []
    for sub_chunk in split_legacy(text[start:end], cfg):
        result.append(
            Chunk(
                content=sub_chunk.content,
                context_header=sub_chunk.context_header,
                start=start + sub_chunk.start,
                end=start + sub_chunk.end,
            )
        )
    return result


def _apply_overlap_aligned(
    text: str,
    cur_end: int,
    boundary_positions: set,
    cfg: SplitterConfig,
) -> int:
    if cfg.chunk_overlap <= 0:
        return cur_end

    target = max(0, cur_end - cfg.chunk_overlap)
    window_start = max(0, cur_end - 2 * cfg.chunk_overlap)
    semantic = [
        pos
        for pos in boundary_positions
        if window_start <= pos < cur_end and pos not in {0, cur_end}
    ]
    if semantic:
        return max(semantic)

    newline = text.rfind("\n", window_start, target + 1)
    if newline != -1:
        return newline + 1
    return target
