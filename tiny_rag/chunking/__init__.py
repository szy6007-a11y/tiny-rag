"""Document chunking primitives for Tiny RAG."""

from .strategy import split, split_parent_child, split_with_diagnostics
from .types import (
    ChildChunk,
    Chunk,
    Diagnostics,
    DocumentProfile,
    ParentChildResult,
    RejectedTier,
    SplitterConfig,
)

__all__ = [
    "ChildChunk",
    "Chunk",
    "Diagnostics",
    "DocumentProfile",
    "ParentChildResult",
    "RejectedTier",
    "SplitterConfig",
    "split",
    "split_parent_child",
    "split_with_diagnostics",
]

