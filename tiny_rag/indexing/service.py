from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Mapping, Sequence

from tiny_rag.embedding import Embedder, batch_embed_with_backoff, sanitize_for_embedding

from .models import (
    BatchIndexStats,
    BatchSaveStats,
    IndexInfo,
    KEYWORDS_RETRIEVER_TYPE,
    VECTOR_RETRIEVER_TYPE,
)
from .repositories.base import IndexRepository


VECTOR_SAVE_BATCH_SIZE = 40
KEYWORD_ONLY_SAVE_BATCH_SIZE = 10
MAX_BATCH_SAVE_CONCURRENCY = 5


class KeywordsVectorHybridIndexer:
    def __init__(self, repository: IndexRepository):
        self.repository = repository

    def batch_index(
        self,
        embedder: Embedder | None,
        index_info_list: Sequence[IndexInfo],
        retriever_types: Sequence[str],
    ) -> BatchIndexStats:
        requested_count = len(index_info_list)
        deduped = deduplicate_by_source_id(index_info_list)
        deduplicated_count = requested_count - len(deduped)
        if not deduped:
            return BatchIndexStats(
                requested_count=requested_count,
                deduplicated_count=deduplicated_count,
            )

        if VECTOR_RETRIEVER_TYPE in retriever_types:
            if embedder is None:
                raise ValueError("embedder is required when vector retriever is enabled")
            content_list = [sanitize_for_embedding(info.content) for info in deduped]
            embeddings = batch_embed_with_backoff(embedder, content_list)
            if len(embeddings) != len(deduped):
                raise ValueError(
                    f"BatchEmbed returned {len(embeddings)} embeddings "
                    f"for {len(deduped)} inputs"
                )
            save_stats = self._batch_save_with_embeddings(deduped, embeddings)
            return BatchIndexStats.from_save_stats(
                save_stats,
                requested_count=requested_count,
                indexed_count=len(deduped),
                deduplicated_count=deduplicated_count,
                embedded_count=len(embeddings),
            )

        save_stats = self._batch_save_keyword_only(deduped)
        return BatchIndexStats.from_save_stats(
            save_stats,
            requested_count=requested_count,
            indexed_count=len(deduped),
            deduplicated_count=deduplicated_count,
            embedded_count=0,
        )

    def _batch_save_with_embeddings(
        self,
        index_info_list: Sequence[IndexInfo],
        embeddings: Sequence[Sequence[float]],
    ) -> BatchSaveStats:
        batches = chunk_sequence(index_info_list, VECTOR_SAVE_BATCH_SIZE)
        return self._concurrent_save(
            batches,
            lambda batch, batch_index: {
                info.source_id: embeddings[batch_index * VECTOR_SAVE_BATCH_SIZE + index]
                for index, info in enumerate(batch)
            },
        )

    def _batch_save_keyword_only(self, index_info_list: Sequence[IndexInfo]) -> BatchSaveStats:
        return self._concurrent_save(
            chunk_sequence(index_info_list, KEYWORD_ONLY_SAVE_BATCH_SIZE),
            lambda _batch, _batch_index: None,
        )

    def _concurrent_save(
        self,
        batches: Sequence[Sequence[IndexInfo]],
        embeddings_for_batch,
    ) -> BatchSaveStats:
        if not batches:
            return BatchSaveStats()

        results: list[BatchSaveStats | None] = [None] * len(batches)

        def save_batch(batch_index: int, batch: Sequence[IndexInfo]) -> None:
            embeddings = embeddings_for_batch(batch, batch_index)
            results[batch_index] = self.repository.batch_save(
                batch,
                embeddings_by_source_id=embeddings,
            )

        workers = min(MAX_BATCH_SAVE_CONCURRENCY, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(save_batch, batch_index, batch)
                for batch_index, batch in enumerate(batches)
            ]
            done, pending = wait(futures, return_when=FIRST_EXCEPTION)
            first_error = next((future.exception() for future in done if future.exception()), None)
            if first_error is not None:
                for future in pending:
                    future.cancel()
                raise first_error
            for future in pending:
                future.result()

        stats = BatchSaveStats()
        for result in results:
            if result is not None:
                stats.add(result)
        return stats

    def estimate_storage_size(
        self,
        embedder: Embedder | None,
        index_info_list: Sequence[IndexInfo],
        retriever_types: Sequence[str],
    ) -> int:
        params: Mapping[str, Sequence[float]] | None = None
        if VECTOR_RETRIEVER_TYPE in retriever_types and embedder is not None:
            dimensions = embedder.get_dimensions()
            params = {info.chunk_id: [0.0] * dimensions for info in index_info_list}
        return self.repository.estimate_storage_size(
            index_info_list,
            embeddings_by_source_id=params,
        )


def deduplicate_by_source_id(index_info_list: Sequence[IndexInfo]) -> list[IndexInfo]:
    seen: set[str] = set()
    deduped: list[IndexInfo] = []
    for info in index_info_list:
        if info.source_id in seen:
            continue
        seen.add(info.source_id)
        deduped.append(info)
    return deduped


def chunk_sequence(items: Sequence[IndexInfo], size: int) -> list[list[IndexInfo]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def normalize_retriever_types(retriever_types: Sequence[str] | None) -> tuple[str, ...]:
    if retriever_types is None:
        return (KEYWORDS_RETRIEVER_TYPE, VECTOR_RETRIEVER_TYPE)
    return tuple(retriever_types)
