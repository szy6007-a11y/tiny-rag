from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable

from tiny_rag.indexing.repositories.sqlite import tokenize_cjk_bigram


def clamp_float(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def clamp01(value: float) -> float:
    if math.isnan(value) or value <= 0 or value == -math.inf:
        return 0.0
    if value >= 1 or value == math.inf:
        return 1.0
    return value


def build_content_signature(content: str) -> str:
    normalized = " ".join(content.strip().lower().split())
    if normalized == "":
        return ""
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def is_all_punct(value: str) -> bool:
    return all(not ch.isalnum() and not ("\u4e00" <= ch <= "\u9fff") for ch in value)


def tokenize_simple(text: str) -> set[str]:
    text = text.strip().lower()
    if text == "":
        return set()
    if contains_chinese(text):
        words = tokenize_cjk_bigram(text).split()
    else:
        words = text.split()
    return {word.strip() for word in words if len(word.strip()) > 1 and not is_all_punct(word)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if len(a) > len(b):
        return jaccard(b, a)
    inter = sum(1 for item in a if item in b)
    union = len(a) + len(b) - inter
    if union == 0:
        return 0.0
    return inter / union


def normalize_content(content: str) -> str:
    return " ".join(content.strip().lower().split())


def is_content_contained(normalized_short: str, normalized_long: str) -> bool:
    if normalized_short == "" or normalized_long == "":
        return False
    if len(normalized_short) > len(normalized_long):
        return False
    return normalized_short in normalized_long


def content_overlap_ratio(a: str, b: str) -> float:
    tokens_a = tokenize_simple(a)
    tokens_b = tokenize_simple(b)
    if not tokens_a or not tokens_b:
        return 0.0
    small, large = (tokens_a, tokens_b)
    if len(tokens_a) > len(tokens_b):
        small, large = tokens_b, tokens_a
    inter = sum(1 for token in small if token in large)
    return inter / len(small)


MIN_OVERLAP_RUNES = 12
DEFAULT_SEARCH_SPAN = 400


def append_with_overlap(acc: str, next_text: str, position_overlap: int) -> str:
    if acc == "":
        return next_text
    if next_text == "":
        return acc

    span = max(position_overlap, 0)
    max_k = min(len(acc), len(next_text))
    cap = max(span * 3, DEFAULT_SEARCH_SPAN)
    if max_k > cap:
        max_k = cap
    head_slack = max(span * 2, 320)

    for k in range(max_k, MIN_OVERLAP_RUNES - 1, -1):
        needle = acc[-k:]
        pos = index_with_max_start(next_text, needle, head_slack)
        if pos >= 0:
            return acc + next_text[pos + k :]
    return acc + next_text


def index_with_max_start(haystack: str, needle: str, max_start: int) -> int:
    if needle == "" or len(needle) > len(haystack):
        return -1
    limit = len(haystack) - len(needle)
    if max_start < limit:
        limit = max_start
    for pos in range(limit + 1):
        if haystack[pos : pos + len(needle)] == needle:
            return pos
    return -1


def concat_no_overlap(a: str, b: str) -> str:
    if a == "":
        return b
    if b == "":
        return a
    max_overlap = min(len(a), len(b))
    for k in range(max_overlap, 0, -1):
        if a[-k:] == b[:k]:
            return a + b[k:]
    return a + b


def merge_ordered_content(prev: str, base: str, next_text: str, max_len: int) -> str:
    content = base
    if prev:
        content = concat_no_overlap(prev, content)
    if next_text:
        content = concat_no_overlap(content, next_text)
    if len(content) > max_len:
        return content[:max_len]
    return content


MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def slice_content_by_document_range(
    content: str, content_start_at: int, range_start: int, range_end: int
) -> str:
    if content == "":
        return content
    rel_start = max(range_start - content_start_at, 0)
    rel_end = min(range_end - content_start_at, len(content))
    if rel_start >= rel_end:
        return ""
    return content[rel_start:rel_end]


def image_urls_in_content(content: str) -> set[str]:
    return {match.group(2) for match in MARKDOWN_IMAGE_RE.finditer(content) if match.group(2)}


def filter_image_info_by_content_urls(content: str, image_info_json: str) -> str:
    if image_info_json == "":
        return ""
    infos = _load_image_infos(image_info_json)
    if not infos:
        return ""
    urls = image_urls_in_content(content)
    if not urls:
        return ""
    filtered = [
        info
        for info in infos
        if str(info.get("url", "")) in urls or str(info.get("original_url", "")) in urls
    ]
    return _dump_image_infos(filtered)


def filter_image_info_by_match_range(
    parent_content: str,
    parent_start_at: int,
    match_start: int,
    match_end: int,
    image_info_json: str,
) -> str:
    if image_info_json == "":
        return ""
    window = slice_content_by_document_range(parent_content, parent_start_at, match_start, match_end)
    return filter_image_info_by_content_urls(window, image_info_json)


def prune_markdown_images_outside_range(
    content: str, content_start_at: int, match_start: int, match_end: int
) -> str:
    matches = list(MARKDOWN_IMAGE_RE.finditer(content))
    if not matches:
        return content
    out: list[str] = []
    last = 0
    for match in matches:
        doc_start = content_start_at + len(content[: match.start()])
        doc_end = content_start_at + len(content[: match.end()])
        in_range = doc_start < match_end and doc_end > match_start
        if in_range:
            out.append(content[last : match.end()])
        else:
            out.append(content[last : match.start()])
        last = match.end()
    out.append(content[last:])
    return collapse_blank_lines("".join(out))


def collapse_blank_lines(text: str) -> str:
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def merge_image_info_json(target_json: str, source_json: str) -> str:
    if source_json == "":
        return target_json
    source = _load_image_infos(source_json)
    if not source:
        return target_json
    target = _load_image_infos(target_json) if target_json else []
    combined = target + source
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for info in combined:
        key = str(info.get("url", ""))
        if key and key not in seen:
            seen.add(key)
            unique.append(info)
    return _dump_image_infos(unique)


def _load_image_infos(raw: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dump_image_infos(infos: Iterable[dict[str, Any]]) -> str:
    items = list(infos)
    if not items:
        return ""
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


RE_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^()\s]*(?:\([^)]*\)[^()\s]*)*\)")
RE_LINKED_IMAGE = re.compile(
    r"\[!\[([^\]]*)\]\(([^()\s]*(?:\([^)]*\)[^()\s]*)*)\)\]"
    r"\([^()\s]*(?:\([^)]*\)[^()\s]*)*\)"
)
RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^()\s]*(?:\([^)]*\)[^()\s]*)*\)")
RE_RAW_URL = re.compile(r"https?://[^\s)\]>]+")
RE_CODE_BLOCK = re.compile(r"```(?:\w*)\n?.*?```", re.S)
RE_LATEX_BLOCK = re.compile(r"\$\$.*?\$\$", re.S)
RE_TABLE_SEP = re.compile(r"^[ \t]*\|[ \t:|-]+\|[ \t]*$", re.M)
RE_TABLE_ROW = re.compile(r"^[ \t]*\|(.+?)\|[ \t]*$", re.M)
RE_HEADING_PREFIX = re.compile(r"^#{1,6}\s+", re.M)
RE_BLOCKQUOTE = re.compile(r"^>\s?", re.M)
RE_BOLD_ITALIC3 = re.compile(r"\*{3}(.+?)\*{3}")
RE_BOLD_ITALIC2 = re.compile(r"\*{2}(.+?)\*{2}")
RE_BOLD_ITALIC1 = re.compile(r"\*(.+?)\*")
RE_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
RE_LIST_MARKER = re.compile(r"^[\t ]*(?:[-*+]|\d+\.)\s+", re.M)
RE_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def clean_passage_for_rerank(text: str) -> str:
    text = RE_CODE_BLOCK.sub("", text)
    text = RE_LATEX_BLOCK.sub("", text)
    text = RE_HTML_TAG.sub("", text)
    text = RE_LINKED_IMAGE.sub(r"![\1](\2)", text)
    text = RE_MARKDOWN_IMAGE.sub("", text)
    text = RE_MARKDOWN_LINK.sub(r"\1", text)
    text = RE_RAW_URL.sub("", text)
    text = RE_TABLE_SEP.sub("", text)

    def table_repl(match: re.Match[str]) -> str:
        cells = [cell.strip() for cell in match.group(1).split("|")]
        return ", ".join(cell for cell in cells if cell)

    text = RE_TABLE_ROW.sub(table_repl, text)
    text = RE_HEADING_PREFIX.sub("", text)
    text = RE_BLOCKQUOTE.sub("", text)
    text = RE_BOLD_ITALIC3.sub(r"\1", text)
    text = RE_BOLD_ITALIC2.sub(r"\1", text)
    text = RE_BOLD_ITALIC1.sub(r"\1", text)
    text = RE_LIST_MARKER.sub("", text)
    text = RE_EXCESSIVE_NEWLINES.sub("\n\n", text)
    return text.strip()
