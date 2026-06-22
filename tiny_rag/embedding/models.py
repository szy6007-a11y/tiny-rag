from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from tiny_rag.cancellation import CancellationToken


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


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "text-embedding-3-small"
    api_key: str = ""
    truncate_prompt_tokens: int = 511
    include_truncate_prompt_tokens: bool = False
    dimensions: int = 1536
    supports_dimension_override: bool = False
    model_id: str = ""
    custom_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        provider = _env_str("EMBEDDING_PROVIDER", "openai")
        if provider.lower() == "siliconflow":
            default_base_url = "https://api.siliconflow.cn/v1"
            default_model_name = "BAAI/bge-m3"
            default_dimensions = 1024
        else:
            default_base_url = "https://api.openai.com/v1"
            default_model_name = "text-embedding-3-small"
            default_dimensions = 1536
        return cls(
            provider=provider,
            base_url=_env_str("EMBEDDING_BASE_URL", default_base_url),
            model_name=_env_str("EMBEDDING_MODEL_NAME", default_model_name),
            api_key=_env_str("EMBEDDING_API_KEY", os.getenv("SILICONFLOW_API_KEY", "")),
            truncate_prompt_tokens=_env_int("TRUNCATE_PROMPT_TOKENS", 511),
            include_truncate_prompt_tokens=_env_bool(
                "EMBEDDING_INCLUDE_TRUNCATE_PROMPT_TOKENS",
                False,
            ),
            dimensions=_env_int("EMBEDDING_DIMENSIONS", default_dimensions),
            supports_dimension_override=_env_bool(
                "EMBEDDING_SUPPORTS_DIMENSION_OVERRIDE",
                False,
            ),
            model_id=os.getenv("EMBEDDING_MODEL_ID", ""),
        )


class Embedder(Protocol):
    def embed(
        self,
        text: str,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[float]:
        ...

    def batch_embed(
        self,
        texts: list[str],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[list[float]]:
        ...

    def get_model_name(self) -> str:
        ...

    def get_dimensions(self) -> int:
        ...

    def get_model_id(self) -> str:
        ...
