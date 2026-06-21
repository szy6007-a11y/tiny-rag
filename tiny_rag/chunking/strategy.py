from __future__ import annotations

from typing import List, Tuple

from .heading import split_heading
from .heuristic import split_heuristic
from .legacy import split_legacy
from .profile import profile_document, select_strategy
from .types import ChildChunk, Chunk, Diagnostics, ParentChildResult, RejectedTier, SplitterConfig
from .utils import assign_sequence


def split(text: str, cfg: SplitterConfig = None) -> List[Chunk]:
    chunks, _ = split_with_diagnostics(text, cfg or SplitterConfig())
    return chunks


def split_with_diagnostics(
    text: str,
    cfg: SplitterConfig = None,
) -> Tuple[List[Chunk], Diagnostics]:
    cfg = (cfg or SplitterConfig()).normalized()
    diagnostics = Diagnostics()
    if not text:
        diagnostics.tier_chain = _tier_chain(cfg, None)
        diagnostics.selected_tier = diagnostics.tier_chain[-1] if diagnostics.tier_chain else ""
        return [], diagnostics

    profile = profile_document(text, cfg.languages) if _uses_auto_router(cfg.strategy) else None
    tiers = _tier_chain(cfg, profile)
    diagnostics.profile = profile
    diagnostics.tier_chain = tiers

    last_chunks: List[Chunk] = []
    for index, tier in enumerate(tiers):
        chunks = assign_sequence(_run_tier(tier, text, cfg, profile))
        last_chunks = chunks
        ok, reason = validate_chunks(chunks, len(text), cfg)
        is_last = index == len(tiers) - 1
        if ok or is_last:
            diagnostics.selected_tier = tier
            if not ok:
                diagnostics.rejected.append(RejectedTier(tier=tier, reason=reason))
            return chunks, diagnostics
        diagnostics.rejected.append(RejectedTier(tier=tier, reason=reason))

    diagnostics.selected_tier = tiers[-1] if tiers else ""
    return assign_sequence(last_chunks), diagnostics


def split_parent_child(
    text: str,
    parent_cfg: SplitterConfig = None,
    child_cfg: SplitterConfig = None,
) -> ParentChildResult:
    if not text:
        return ParentChildResult(parents=[], children=[])

    parent_cfg = (parent_cfg or SplitterConfig(chunk_size=4096)).normalized()
    child_cfg = (child_cfg or SplitterConfig(chunk_size=384)).normalized()
    parents = split(text, parent_cfg)
    if not parents:
        return ParentChildResult(parents=[], children=[])

    kept_parents: List[Chunk] = []
    children: List[ChildChunk] = []

    for parent in parents:
        local_children = split(parent.content, child_cfg)
        parent_index = -1
        if len(local_children) > 1 or (
            len(local_children) == 1 and local_children[0].content != parent.content
        ):
            parent_index = len(kept_parents)
            kept_parents.append(parent)

        for local in local_children:
            context_header = _merge_breadcrumbs(parent.context_header, local.context_header)
            child = ChildChunk(
                content=local.content,
                context_header=context_header,
                seq=len(children),
                start=parent.start + local.start,
                end=parent.start + local.end,
                parent_index=parent_index,
            )
            children.append(child)

    assign_sequence(kept_parents)
    return ParentChildResult(parents=kept_parents, children=children)


def validate_chunks(chunks: List[Chunk], total_chars: int, cfg: SplitterConfig) -> Tuple[bool, str]:
    cfg = cfg.normalized(default_overlap=False)
    if not chunks:
        return False, "no chunks produced"

    if total_chars > 2 * cfg.chunk_size and len(chunks) == 1:
        return False, "single chunk for large document"

    lengths = [len(chunk.content) for chunk in chunks]
    max_len = max(lengths)
    tiny_count = sum(1 for length in lengths[:-1] if length < 50)
    if tiny_count > len(chunks) / 4 and tiny_count > 2:
        return False, "too many tiny chunks"

    if max_len < cfg.chunk_size / 4 and total_chars > cfg.chunk_size:
        return False, "all chunks far below target size"

    if max_len > 2 * cfg.chunk_size:
        return False, "chunk exceeds 2x target size"

    for chunk in chunks:
        if chunk.start < 0 or chunk.end < chunk.start or chunk.end > total_chars:
            return False, "chunk offsets out of range"
        if chunk.end - chunk.start > len(chunk.content):
            return False, "chunk offsets exceed content length"

    return True, ""


def _tier_chain(cfg: SplitterConfig, profile) -> List[str]:
    strategy = (cfg.strategy or "").lower()
    if strategy == "heading":
        return ["heading", "legacy"]
    if strategy == "heuristic":
        return ["heuristic", "legacy"]
    if strategy in {"", "legacy", "recursive"}:
        return ["legacy"]
    if _uses_auto_router(strategy):
        return select_strategy(profile) if profile else ["legacy"]
    return ["legacy"]


def _uses_auto_router(strategy: str) -> bool:
    strategy = (strategy or "").lower()
    return strategy == "auto" or strategy not in {"", "heading", "heuristic", "legacy", "recursive"}


def _run_tier(tier: str, text: str, cfg: SplitterConfig, profile=None) -> List[Chunk]:
    if tier == "heading":
        return split_heading(text, cfg, profile)
    if tier == "heuristic":
        return split_heuristic(text, cfg)
    return split_legacy(text, cfg)


def _merge_breadcrumbs(parent: str, child: str) -> str:
    if not parent:
        return child
    if not child:
        return parent

    parent_lines = parent.split("\n")
    child_lines = child.split("\n")
    if (
        parent_lines
        and child_lines
        and parent_lines[-1].strip() == child_lines[0].strip()
    ):
        child_lines = child_lines[1:]

    if not child_lines:
        return parent
    return f"{parent}\n" + "\n".join(child_lines)
