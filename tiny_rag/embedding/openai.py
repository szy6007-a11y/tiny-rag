from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .batch import batch_embed_with_pool
from .models import Embedder, EmbeddingConfig


RESERVED_HEADER_NAMES = {"authorization", "content-type"}


@dataclass
class OpenAIEmbedder(Embedder):
    api_key: str
    base_url: str
    model_name: str
    truncate_prompt_tokens: int = 511
    dimensions: int = 0
    model_id: str = ""
    supports_dimension_override: bool = False
    custom_headers: dict[str, str] | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "OpenAIEmbedder":
        if config.provider.lower() != "openai":
            raise ValueError(f"unsupported embedding provider: {config.provider}")
        return cls(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            truncate_prompt_tokens=config.truncate_prompt_tokens,
            dimensions=config.dimensions,
            model_id=config.model_id,
            supports_dimension_override=config.supports_dimension_override,
            custom_headers=config.custom_headers,
        )

    @classmethod
    def from_env(cls) -> "OpenAIEmbedder":
        return cls.from_config(EmbeddingConfig.from_env())

    def __post_init__(self) -> None:
        if self.base_url == "":
            self.base_url = "https://api.openai.com/v1"
        self.base_url = self.base_url.rstrip("/")
        if self.model_name == "":
            raise ValueError("model name is required")
        if self.truncate_prompt_tokens == 0:
            self.truncate_prompt_tokens = 511

    def set_custom_headers(self, headers: dict[str, str]) -> None:
        self.custom_headers = headers

    def set_supports_dimension_override(self, supported: bool) -> None:
        self.supports_dimension_override = supported

    def embed(self, text: str) -> list[float]:
        for _ in range(3):
            embeddings = self.batch_embed([text])
            if embeddings:
                return embeddings[0]
        raise RuntimeError("no embedding returned")

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
            "truncate_prompt_tokens": self.truncate_prompt_tokens,
        }
        if self.supports_dimensions_param():
            body["dimensions"] = self.dimensions

        response = self._post_with_retry(body)
        if response.status_code != 200:
            body_text = response.text
            if len(body_text) > 1000:
                body_text = body_text[:1000] + "... (truncated)"
            raise RuntimeError(
                f"EmbedBatch API error: Http Status {response.status_code}, "
                f"Response: {body_text}"
            )

        payload = response.json()
        return [item["embedding"] for item in payload.get("data", [])]

    def batch_embed_with_pool(
        self,
        model: Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        return batch_embed_with_pool(model, texts)

    def supports_dimensions_param(self) -> bool:
        return self.supports_dimension_override and self.dimensions > 0

    def get_model_name(self) -> str:
        return self.model_name

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_model_id(self) -> str:
        return self.model_id

    def _post_with_retry(self, json_body: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        url = self.base_url + "/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        for key, value in (self.custom_headers or {}).items():
            if key.lower() not in RESERVED_HEADER_NAMES:
                headers[key] = value

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries + 1):
                if attempt > 0:
                    time.sleep(min(2 ** (attempt - 1), 10))
                try:
                    return client.post(url, headers=headers, json=json_body)
                except Exception as exc:
                    last_error = exc

        if last_error is not None:
            raise RuntimeError(f"send request: {last_error}") from last_error
        raise RuntimeError("send request failed")
