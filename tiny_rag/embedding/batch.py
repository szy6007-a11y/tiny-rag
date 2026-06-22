from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Sequence, TypeVar

from .models import Embedder


T = TypeVar("T")
DEFAULT_BATCH_EMBED_SIZE = 5
DEFAULT_CONCURRENCY_POOL_SIZE = 5
EMBED_RETRY_ATTEMPTS = 5
EMBED_RETRY_BASE_DELAY_SECONDS = 0.2


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def batch_embed_size() -> int:
    size = _env_int("BATCH_EMBED_SIZE", DEFAULT_BATCH_EMBED_SIZE)
    if size <= 0:
        raise ValueError("BATCH_EMBED_SIZE must be positive")
    return size


def concurrency_pool_size() -> int:
    size = _env_int("CONCURRENCY_POOL_SIZE", DEFAULT_CONCURRENCY_POOL_SIZE)
    if size <= 0:
        raise ValueError("CONCURRENCY_POOL_SIZE must be positive")
    return size


def chunk_sequence(items: Sequence[T], size: int) -> list[list[T]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def batch_embed_with_pool(
    model: Embedder,
    texts: list[str],
    *,
    batch_size: int | None = None,
    max_workers: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    resolved_batch_size = batch_size or batch_embed_size()
    chunks = chunk_sequence(texts, resolved_batch_size)
    results: list[list[float] | None] = [None] * len(texts)

    def process(batch_index: int, batch: list[str]) -> None:
        embeddings = model.batch_embed(batch)
        if len(embeddings) != len(batch):
            raise ValueError(
                f"BatchEmbed returned {len(embeddings)} embeddings for {len(batch)} inputs"
            )
        offset = batch_index * resolved_batch_size
        for index, embedding in enumerate(embeddings):
            results[offset + index] = embedding

    workers = min(max_workers or concurrency_pool_size(), len(chunks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process, batch_index, batch)
            for batch_index, batch in enumerate(chunks)
        ]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        first_error = next((future.exception() for future in done if future.exception()), None)
        if first_error is not None:
            for future in pending:
                future.cancel()
            raise first_error
        for future in pending:
            future.result()

    return [embedding for embedding in results if embedding is not None]


def batch_embed_with_backoff(model: Embedder, texts: list[str]) -> list[list[float]]:
    delay = EMBED_RETRY_BASE_DELAY_SECONDS
    last_error: Exception | None = None

    for attempt in range(EMBED_RETRY_ATTEMPTS):
        try:
            batch_with_pool = getattr(model, "batch_embed_with_pool", None)
            if callable(batch_with_pool):
                return batch_with_pool(model, texts)
            return batch_embed_with_pool(model, texts)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < EMBED_RETRY_ATTEMPTS:
                time.sleep(delay)
                delay *= 2

    if last_error is not None:
        raise last_error
    return []
