from __future__ import annotations

import re


NUMBERED_HEADING_RE = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+){1,3}\.?|(?:\d+|[IVX]{1,5})\.)[ \t]+\S.{0,200}$"
)
ROMAN_HEADING_RE = re.compile(
    r"^\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)[.)]?\s+[A-Z][^\n]{0,140}$"
)
ENGLISH_CHAPTER_RE = re.compile(
    r"^[ \t]*(?:Chapter|Section|Part)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$",
    re.IGNORECASE,
)
GERMAN_CHAPTER_RE = re.compile(
    r"^[ \t]*(?:Kapitel|Abschnitt|Teil)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$",
    re.IGNORECASE,
)
CHINESE_CHAPTER_RE = re.compile(
    r"^[ \t]*第[ \t]*[一二三四五六七八九十百千零〇0-9]+[ \t]*(?:章|节|節|部分|篇)[ \t]?.{0,200}$"
)
VISUAL_SEPARATOR_RE = re.compile(r"^\s*[-=_*]{3,}\s*$")
PAGE_FOOTER_RE = re.compile(
    r"^[ \t]*(?:Seite|Page|页码?)\s+\d+(?:\s*(?:von|of|/)\s*\d+)?[ \t]*$",
    re.IGNORECASE,
)
BLANK_BLOCK_RE = re.compile(r"\n{3,}")
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def is_all_caps_heading(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 80):
        return False
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) < 3:
        return False
    return not any(char.islower() for char in letters)


def is_table_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and len(stripped.split("|")) >= 3


def chapter_marker_kind(line: str, languages=None) -> str:
    enabled = _enabled_chapter_languages(languages or [])
    stripped = line.strip()
    if "en" in enabled and ENGLISH_CHAPTER_RE.match(stripped):
        return "en"
    if "de" in enabled and GERMAN_CHAPTER_RE.match(stripped):
        return "de"
    if "zh" in enabled and CHINESE_CHAPTER_RE.match(stripped):
        return "zh"
    return ""


def _enabled_chapter_languages(languages) -> set:
    if not languages:
        return {"en", "de", "zh"}

    aliases = {
        "en": "en",
        "eng": "en",
        "english": "en",
        "de": "de",
        "deu": "de",
        "ger": "de",
        "german": "de",
        "deutsch": "de",
        "zh": "zh",
        "zho": "zh",
        "chi": "zh",
        "chinese": "zh",
        "cn": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
    }
    enabled = {aliases[lang.lower()] for lang in languages if lang.lower() in aliases}
    return enabled or {"en", "de", "zh"}
