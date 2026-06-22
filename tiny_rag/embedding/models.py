from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "text-embedding-3-small"
    api_key: str = ""
    truncate_prompt_tokens: int = 511
    dimensions: int = 1536
    supports_dimension_override: bool = False
    model_id: str = ""
    custom_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            truncate_prompt_tokens=_env_int("TRUNCATE_PROMPT_TOKENS", 511),
            dimensions=_env_int("EMBEDDING_DIMENSIONS", 1536),
            supports_dimension_override=_env_bool(
                "EMBEDDING_SUPPORTS_DIMENSION_OVERRIDE",
                False,
            ),
            model_id=os.getenv("EMBEDDING_MODEL_ID", ""),
        )


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def get_model_name(self) -> str:
        ...

    def get_dimensions(self) -> int:
        ...

    def get_model_id(self) -> str:
        ...
