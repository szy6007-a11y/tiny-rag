from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    start: int
    end: int

    @property
    def markdown(self) -> str:
        return f"{'#' * self.level} {self.title}"


def iter_lines_with_offsets(text: str) -> Iterator[Tuple[str, int, int]]:
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        yield line, start, end
        start = end
    if start < len(text):
        yield text[start:], start, len(text)


def is_fence_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def find_markdown_headings(text: str) -> List[Heading]:
    headings: List[Heading] = []
    in_fence = False
    for line, start, end in iter_lines_with_offsets(text):
        if is_fence_line(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line.rstrip("\r\n"))
        if match:
            title = match.group(2).strip().strip("#").strip()
            headings.append(Heading(len(match.group(1)), title, start, end))
    return headings


def choose_primary_heading_level(headings: List[Heading]) -> int:
    counts = {}
    for heading in headings:
        counts[heading.level] = counts.get(heading.level, 0) + 1
    frequent = [level for level, count in counts.items() if count >= 3]
    if frequent:
        return min(frequent)
    return max(counts)


def assign_sequence(chunks):
    for index, chunk in enumerate(chunks):
        chunk.seq = index
    return chunks


def common_line_prefix(left: str, right: str) -> List[str]:
    left_lines = [line for line in left.splitlines() if line.strip()]
    right_lines = [line for line in right.splitlines() if line.strip()]
    prefix: List[str] = []
    for left_line, right_line in zip(left_lines, right_lines):
        if left_line != right_line:
            break
        prefix.append(left_line)
    return prefix

