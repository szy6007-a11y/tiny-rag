from __future__ import annotations

import logging
import re


SAFETY_MAX_CHARS = 20_000


EMBEDDING_IMAGE_PAYLOAD_PATTERNS = [
    re.compile(
        r"""<img\b[^>]*\bsrc=["']\s*data:image/[a-z0-9.+-]+;base64,[^"']+["'][^>]*>""",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"""!\[[^\]]*\]\(\s*data:image/[a-z0-9.+-]+;base64,[^)]+\)""",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"""data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=]{200,}""",
        re.IGNORECASE,
    ),
    re.compile(
        r"""data:[a-z0-9.+/-]+;base64,[a-z0-9+/=]{200,}""",
        re.IGNORECASE,
    ),
]


def sanitize_for_embedding(content: str) -> str:
    sanitized = content
    if "base64," in content:
        for pattern in EMBEDDING_IMAGE_PAYLOAD_PATTERNS:
            sanitized = pattern.sub("[image]", sanitized)

    if len(sanitized) <= SAFETY_MAX_CHARS:
        return sanitized

    logging.getLogger(__name__).warning(
        "embedding input truncated: %d runes -> %d",
        len(sanitized),
        SAFETY_MAX_CHARS,
    )
    return sanitized[:SAFETY_MAX_CHARS]
