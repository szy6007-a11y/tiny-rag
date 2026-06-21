from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from .patterns import TABLE_SEPARATOR_RE, is_table_row
from .utils import is_fence_line, iter_lines_with_offsets

Span = Tuple[int, int]

LATEX_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
LINK_RE = re.compile(r"(?<!!)\[[^\]]+]\([^)]+\)")


def protected_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    spans.extend(_regex_spans(LATEX_BLOCK_RE, text))
    spans.extend(_regex_spans(IMAGE_RE, text))
    spans.extend(_regex_spans(LINK_RE, text))
    spans.extend(_fenced_code_spans(text))
    spans.extend(_table_spans(text))
    return merge_spans(spans)


def merge_spans(spans: Iterable[Span]) -> List[Span]:
    sorted_spans = sorted((start, end) for start, end in spans if start < end)
    if not sorted_spans:
        return []
    merged: List[Span] = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def is_inside_spans(pos: int, spans: List[Span]) -> bool:
    return any(start < pos < end for start, end in spans)


def _regex_spans(pattern: re.Pattern, text: str) -> List[Span]:
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _fenced_code_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    in_fence = False
    fence_start = 0
    for line, start, end in iter_lines_with_offsets(text):
        if is_fence_line(line):
            if not in_fence:
                in_fence = True
                fence_start = start
            else:
                spans.append((fence_start, end))
                in_fence = False
    if in_fence:
        spans.append((fence_start, len(text)))
    return spans


def _table_spans(text: str) -> List[Span]:
    lines = list(iter_lines_with_offsets(text))
    spans: List[Span] = []
    index = 0
    while index < len(lines):
        line, start, _ = lines[index]
        if not is_table_row(line):
            index += 1
            continue

        group_start = index
        has_separator = bool(TABLE_SEPARATOR_RE.match(line.strip()))
        index += 1
        while index < len(lines) and is_table_row(lines[index][0]):
            has_separator = has_separator or bool(TABLE_SEPARATOR_RE.match(lines[index][0].strip()))
            index += 1

        if has_separator and index - group_start >= 2:
            spans.append((start, lines[index - 1][2]))
    return spans

