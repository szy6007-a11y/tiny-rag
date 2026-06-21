from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_SEPARATORS = ["\n\n", "\n", "。"]
ABSOLUTE_MAX_SIZE = 7500


@dataclass
class SplitterConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: List[str] = field(default_factory=lambda: list(DEFAULT_SEPARATORS))
    strategy: str = ""
    token_limit: int = 0
    languages: List[str] = field(default_factory=list)

    def normalized(self, default_overlap: bool = True) -> "SplitterConfig":
        chunk_size = self.chunk_size if self.chunk_size > 0 else DEFAULT_CHUNK_SIZE

        chunk_overlap = self.chunk_overlap
        if chunk_overlap <= 0 and (default_overlap or chunk_overlap < 0):
            chunk_overlap = DEFAULT_CHUNK_OVERLAP

        separators = [sep for sep in self.separators if sep] if self.separators else []
        if not separators:
            separators = list(DEFAULT_SEPARATORS)

        if self.token_limit > 0:
            budget = int(self.token_limit * _estimated_chars_per_token(self.languages))
            if budget > 0:
                chunk_size = min(chunk_size, budget)

        max_overlap = max(chunk_size // 2, 0)
        if chunk_overlap > max_overlap:
            chunk_overlap = max_overlap

        return SplitterConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            strategy=(self.strategy or "").strip().lower(),
            token_limit=self.token_limit,
            languages=list(self.languages),
        )


def _estimated_chars_per_token(languages: List[str]) -> float:
    cjk_markers = {"zh", "zho", "chi", "chinese", "cn", "ja", "jpn", "ko", "kor"}
    lowered = {lang.lower() for lang in languages}
    if lowered & cjk_markers:
        return 1.5
    return 4.0


@dataclass
class Chunk:
    content: str
    context_header: str = ""
    seq: int = 0
    start: int = 0
    end: int = 0

    def embedding_content(self) -> str:
        body = self.content.strip()
        if not self.context_header:
            return body
        return f"{self.context_header}\n\n{body}"


@dataclass
class ChildChunk(Chunk):
    parent_index: int = -1


@dataclass
class ParentChildResult:
    parents: List[Chunk]
    children: List[ChildChunk]


@dataclass
class RejectedTier:
    tier: str
    reason: str


@dataclass
class DocumentProfile:
    total_chars: int = 0
    total_lines: int = 0
    heading_count: int = 0
    heading_density: float = 0.0
    primary_heading_level: Optional[int] = None
    form_feed_count: int = 0
    numbered_heading_count: int = 0
    roman_heading_count: int = 0
    chapter_marker_count: int = 0
    all_caps_heading_count: int = 0
    visual_separator_count: int = 0
    page_footer_count: int = 0
    blank_block_count: int = 0
    table_count: int = 0
    code_fence_count: int = 0
    languages: List[str] = field(default_factory=list)

    @property
    def structure_marker_count(self) -> int:
        return (
            self.form_feed_count
            + self.numbered_heading_count
            + self.roman_heading_count
            + self.chapter_marker_count
            + self.all_caps_heading_count
            + self.visual_separator_count
            + self.page_footer_count
            + self.blank_block_count
        )


@dataclass
class Diagnostics:
    selected_tier: str = ""
    tier_chain: List[str] = field(default_factory=list)
    rejected: List[RejectedTier] = field(default_factory=list)
    profile: Optional[DocumentProfile] = None
