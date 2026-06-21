from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

from .patterns import TABLE_SEPARATOR_RE, is_table_row
from .protected import protected_spans
from .types import ABSOLUTE_MAX_SIZE, Chunk, SplitterConfig


MARKDOWN_TABLE_HOOK_PRIORITY = 15
TABLE_HEADER_START_RE = re.compile(
    r"^\s*(?:\|[^|\n]*)+\|?\s*(?:\r?\n)+\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*(?:\r?\n)+$",
    re.IGNORECASE | re.DOTALL,
)
TABLE_HEADER_END_RE = re.compile(r"^\s*$|^\s*[^|\s].*$", re.IGNORECASE | re.DOTALL)
TABLE_ROW_RE = re.compile(r"^\s*(?:\|[^|\n]*)+\|\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Unit:
    start: int
    end: int
    text: str = ""
    synthetic: bool = False

    @property
    def length(self) -> int:
        if self.synthetic:
            return len(self.text)
        return self.end - self.start


@dataclass(frozen=True)
class HeaderTrackerHook:
    start_pattern: re.Pattern
    end_pattern: re.Pattern
    priority: int


DEFAULT_HEADER_HOOKS = [
    HeaderTrackerHook(
        start_pattern=TABLE_HEADER_START_RE,
        end_pattern=TABLE_HEADER_END_RE,
        priority=MARKDOWN_TABLE_HOOK_PRIORITY,
    )
]


def split_legacy(text: str, cfg: SplitterConfig) -> List[Chunk]:
    cfg = cfg.normalized(default_overlap=False)
    if not text:
        return []
    units = _build_units(text, cfg)
    chunks = _merge_units(text, units, cfg)
    return chunks


def _build_units(text: str, cfg: SplitterConfig) -> List[Unit]:
    spans = protected_spans(text)
    units: List[Unit] = []
    cursor = 0
    for start, end in spans:
        if cursor < start:
            units.extend(_split_plain_region(text, cursor, start, cfg.separators, 0, cfg.chunk_size))
        if _is_table_text(text[start:end]):
            units.extend(_split_table_units(text, start, end))
        else:
            units.append(Unit(start, end))
        cursor = end
    if cursor < len(text):
        units.extend(_split_plain_region(text, cursor, len(text), cfg.separators, 0, cfg.chunk_size))
    return [unit for unit in units if unit.start < unit.end]


def _split_plain_region(
    text: str,
    start: int,
    end: int,
    separators: List[str],
    sep_index: int,
    chunk_size: int,
) -> List[Unit]:
    if start >= end:
        return []
    if sep_index >= len(separators):
        return [Unit(start, end)]

    separator = separators[sep_index]
    pieces = _split_keep_separator(text, start, end, separator)
    units: List[Unit] = []
    for piece_start, piece_end, is_separator in pieces:
        if piece_start == piece_end:
            continue
        if is_separator:
            units.append(Unit(piece_start, piece_end))
        elif piece_end - piece_start > chunk_size and sep_index + 1 < len(separators):
            units.extend(
                _split_plain_region(
                    text,
                    piece_start,
                    piece_end,
                    separators,
                    sep_index + 1,
                    chunk_size,
                )
            )
        else:
            units.append(Unit(piece_start, piece_end))
    return units


def _split_keep_separator(text: str, start: int, end: int, separator: str):
    cursor = start
    sep_len = len(separator)
    while cursor < end:
        found = text.find(separator, cursor, end)
        if found == -1:
            yield cursor, end, False
            return
        if found > cursor:
            yield cursor, found, False
        yield found, found + sep_len, True
        cursor = found + sep_len


def _merge_units(text: str, units: List[Unit], cfg: SplitterConfig) -> List[Chunk]:
    if not units:
        return []

    chunks: List[Chunk] = []
    current: List[Unit] = []
    cur_len = 0
    header_tracker = HeaderTracker()

    for unit in units:
        unit_text = _unit_text(text, unit)
        unit_len = len(unit_text)

        if unit_len > ABSOLUTE_MAX_SIZE:
            _flush_units(text, current, chunks)
            header_tracker.update(unit_text)
            current = []
            cur_len = 0
            chunks.extend(_hard_split_unit(text, unit))
            continue

        header_tracker.update(unit_text)
        if header_tracker.header_ended_this_unit and current:
            _flush_units(text, current, chunks)
            current = []
            cur_len = 0

        headers = header_tracker.get_headers()
        headers_len = len(headers)
        if headers_len > cfg.chunk_size:
            headers = ""
            headers_len = 0

        if current and cur_len + unit_len + headers_len > cfg.chunk_size:
            _flush_units(text, current, chunks)
            current = _compute_overlap_units(text, current, cfg, unit_len)
            cur_len = _units_length(current)

            if headers and headers_len + unit_len <= cfg.chunk_size:
                while current and cur_len + unit_len + headers_len > cfg.chunk_size:
                    cur_len -= current[0].length
                    current = current[1:]

                overlap_text = _units_text(text, current)
                if not _header_already_present(headers, overlap_text, unit_text) and not _header_column_mismatch(
                    headers, unit_text
                ):
                    start_pos = current[0].start if current else unit.start
                    header_unit = Unit(start=start_pos, end=start_pos, text=headers, synthetic=True)
                    current = [header_unit] + current
                    cur_len += headers_len

        if current and cur_len + unit_len > ABSOLUTE_MAX_SIZE:
            _flush_units(text, current, chunks)
            current = []
            cur_len = 0

        current.append(unit)
        cur_len += unit_len

    _flush_units(text, current, chunks)
    return chunks


def _flush_units(text: str, units: List[Unit], chunks: List[Chunk]) -> None:
    if not units:
        return
    real_units = [unit for unit in units if not unit.synthetic]
    if not real_units:
        return
    start = real_units[0].start
    end = real_units[-1].end
    content = _units_text(text, real_units)
    synthetic_headers = [unit.text.strip() for unit in units if unit.synthetic and unit.text.strip()]
    context_header = "\n".join(synthetic_headers)
    if content:
        chunks.append(Chunk(content=content, context_header=context_header, start=start, end=end))


def _compute_overlap_units(
    text: str,
    units: List[Unit],
    cfg: SplitterConfig,
    next_len: int,
) -> List[Unit]:
    if cfg.chunk_overlap <= 0:
        return []
    selected: List[Unit] = []
    total = 0
    for unit in reversed(units):
        if unit.synthetic:
            continue
        if total + unit.length > cfg.chunk_overlap:
            break
        if total + unit.length + next_len > cfg.chunk_size:
            break
        selected.insert(0, unit)
        total += unit.length
    return _trim_overlap_leading_noise(text, selected, cfg)


def _trim_overlap_leading_noise(text: str, units: List[Unit], cfg: SplitterConfig) -> List[Unit]:
    trimmed = list(units)
    while trimmed and _is_noise_unit(_unit_text(text, trimmed[0]), cfg):
        trimmed.pop(0)
    if not trimmed:
        return []

    first = trimmed[0]
    if first.synthetic:
        return trimmed

    segment = _unit_text(text, first)
    leading = len(segment) - len(segment.lstrip())
    if leading:
        trimmed[0] = Unit(first.start + leading, first.end)
    return trimmed


def _is_noise_unit(segment: str, cfg: SplitterConfig) -> bool:
    if not segment:
        return True
    if not segment.strip():
        return True
    return segment in cfg.separators


def _units_length(units: List[Unit]) -> int:
    return sum(unit.length for unit in units)


def _hard_split_unit(text: str, unit: Unit) -> List[Chunk]:
    chunks: List[Chunk] = []
    unit_text = _unit_text(text, unit)
    offset = 0
    while offset < len(unit_text):
        chunk_end = min(offset + ABSOLUTE_MAX_SIZE, len(unit_text))
        if chunk_end < len(unit_text):
            natural = _find_natural_break(unit_text, offset, chunk_end)
            if natural > offset:
                chunk_end = natural
        if chunk_end <= offset:
            chunk_end = min(offset + ABSOLUTE_MAX_SIZE, len(unit_text))
        chunks.append(
            Chunk(
                content=unit_text[offset:chunk_end],
                start=unit.start + offset,
                end=unit.start + chunk_end,
            )
        )
        offset = chunk_end
    return chunks


def _find_natural_break(text: str, offset: int, chunk_end: int) -> int:
    window_start = max(offset, chunk_end - 200)
    for index in range(chunk_end - 1, window_start - 1, -1):
        if text[index] in {"\n", " "}:
            return index + 1
    return chunk_end


class HeaderTracker:
    def __init__(self) -> None:
        self.hooks = DEFAULT_HEADER_HOOKS
        self.active_headers: Dict[int, str] = {}
        self.ended_headers: Dict[int, bool] = {}
        self.pending_extend: Dict[int, bool] = {}
        self.pending_table_break = False
        self.table_recently_ended = False
        self.header_ended_this_unit = False

    def update(self, text: str) -> None:
        self.header_ended_this_unit = False
        self._resolve_pending_table_break(text)
        self._end_matching_headers(text)
        self._track_table_break_or_column_change(text)
        self._extend_empty_headers(text)
        self._start_matching_headers(text)

        if not self.active_headers:
            self.ended_headers.clear()

    def get_headers(self) -> str:
        if not self.active_headers:
            return ""
        return "\n".join(
            text
            for _, text in sorted(
                self.active_headers.items(),
                key=lambda item: item[0],
                reverse=True,
            )
        )

    def _resolve_pending_table_break(self, text: str) -> None:
        if not self.pending_table_break:
            return

        self.pending_table_break = False
        if MARKDOWN_TABLE_HOOK_PRIORITY in self.active_headers:
            if _first_table_row_column_count(text) > 0:
                self._clear_table_header()
                self.header_ended_this_unit = True
            else:
                self._clear_table_header()
                self.table_recently_ended = True

    def _end_matching_headers(self, text: str) -> None:
        for hook in self.hooks:
            if hook.priority not in self.active_headers:
                continue
            if hook.end_pattern.search(text):
                self.ended_headers[hook.priority] = True
                del self.active_headers[hook.priority]
                self.pending_extend.pop(hook.priority, None)
                if hook.priority == MARKDOWN_TABLE_HOOK_PRIORITY:
                    self.table_recently_ended = True

    def _track_table_break_or_column_change(self, text: str) -> None:
        if MARKDOWN_TABLE_HOOK_PRIORITY not in self.active_headers:
            return
        if self.pending_extend.get(MARKDOWN_TABLE_HOOK_PRIORITY):
            return
        if _split_ends_with_paragraph_break(text):
            self.pending_table_break = True
        else:
            self._end_table_header_on_column_mismatch(text)

    def _extend_empty_headers(self, text: str) -> None:
        for priority in list(self.pending_extend):
            if priority in self.active_headers and TABLE_ROW_RE.search(text):
                separator = _extract_separator_line(self.active_headers[priority])
                self.active_headers[priority] = text + separator
            self.pending_extend.pop(priority, None)

    def _start_matching_headers(self, text: str) -> None:
        for hook in self.hooks:
            if hook.priority in self.active_headers:
                continue
            match = hook.start_pattern.search(text)
            if self.ended_headers.get(hook.priority):
                can_restart_table = (
                    hook.priority == MARKDOWN_TABLE_HOOK_PRIORITY
                    and self.header_ended_this_unit
                    and bool(match)
                )
                if not can_restart_table:
                    continue

            if not match:
                continue

            if hook.priority == MARKDOWN_TABLE_HOOK_PRIORITY and self.table_recently_ended:
                self.header_ended_this_unit = True
                self.table_recently_ended = False

            header = match.group(0)
            self.active_headers[hook.priority] = header
            if _is_empty_table_header_row(header):
                self.pending_extend[hook.priority] = True

    def _clear_table_header(self) -> None:
        self.ended_headers[MARKDOWN_TABLE_HOOK_PRIORITY] = True
        self.active_headers.pop(MARKDOWN_TABLE_HOOK_PRIORITY, None)
        self.pending_extend.pop(MARKDOWN_TABLE_HOOK_PRIORITY, None)

    def _end_table_header_on_column_mismatch(self, text: str) -> None:
        header = self.active_headers.get(MARKDOWN_TABLE_HOOK_PRIORITY)
        if not header:
            return

        row_cols = _first_table_row_column_count(text)
        header_cols = _header_table_column_count(header)
        if row_cols > 0 and header_cols > 0 and row_cols != header_cols:
            self._clear_table_header()
            self.header_ended_this_unit = True
            self.table_recently_ended = True


def _split_table_units(text: str, start: int, end: int) -> List[Unit]:
    units: List[Unit] = []
    line_ranges = []
    cursor = start
    while cursor < end:
        newline = text.find("\n", cursor, end)
        line_end = end if newline == -1 else newline + 1
        line_ranges.append((cursor, line_end))
        cursor = line_end

    index = 0
    while index < len(line_ranges):
        line_start, line_end = line_ranges[index]
        if index + 1 < len(line_ranges):
            next_start, next_end = line_ranges[index + 1]
            if _is_table_row_text(text[line_start:line_end]) and TABLE_SEPARATOR_RE.match(
                text[next_start:next_end].strip()
            ):
                units.append(Unit(line_start, next_end))
                index += 2
                continue

        units.append(Unit(line_start, line_end))
        index += 1
    return units


def _is_table_text(segment: str) -> bool:
    lines = [line for line in segment.splitlines() if line.strip()]
    return any(TABLE_SEPARATOR_RE.match(line.strip()) for line in lines)


def _unit_text(text: str, unit: Unit) -> str:
    if unit.synthetic:
        return unit.text
    return text[unit.start : unit.end]


def _units_text(text: str, units: List[Unit]) -> str:
    return "".join(_unit_text(text, unit) for unit in units)


def _is_table_row_text(text: str) -> bool:
    return is_table_row(text.strip())


def _header_already_present(headers: str, overlap_text: str, next_text: str) -> bool:
    header = headers.strip()
    if not header:
        return True
    return header in overlap_text.strip() or next_text.startswith(headers) or header in next_text.strip()


def _header_column_mismatch(headers: str, next_text: str) -> bool:
    next_line = next((line for line in next_text.splitlines() if line.strip()), "")
    if not _is_table_row_text(next_line) or TABLE_SEPARATOR_RE.match(next_line.strip()):
        return False

    header_line = next((line for line in headers.splitlines() if _is_table_row_text(line)), "")
    header_cols = _table_column_count(header_line)
    next_cols = _table_column_count(next_line)
    return header_cols > 0 and next_cols > 0 and header_cols != next_cols


def _table_column_count(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    if not stripped:
        return 0
    return len(stripped.split("|"))


def _split_ends_with_paragraph_break(text: str) -> bool:
    return bool(re.search(r"(?:\r?\n){2,}\s*$", text))


def _is_empty_table_header_row(header: str) -> bool:
    first_line = header.splitlines()[0].strip() if header.splitlines() else ""
    return bool(first_line) and all(char in {"|", " ", "\t"} for char in first_line)


def _extract_separator_line(header: str) -> str:
    for line in header.splitlines():
        if "---" in line:
            return line + "\n"
    return ""


def _first_table_row_column_count(text: str) -> int:
    for line in text.splitlines():
        if _is_table_row_text(line) and not TABLE_SEPARATOR_RE.match(line.strip()):
            return _table_column_count(line)
    return 0


def _header_table_column_count(header: str) -> int:
    for line in header.splitlines():
        if _is_table_row_text(line) and not TABLE_SEPARATOR_RE.match(line.strip()):
            return _table_column_count(line)
    return 0
