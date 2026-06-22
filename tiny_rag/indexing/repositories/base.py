from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from tiny_rag.cancellation import CancellationToken
from tiny_rag.indexing.models import BatchSaveStats, IndexInfo, RetrieveParams, RetrieveResult


class IndexRepository(Protocol):
    def engine_type(self) -> str:
        ...

    def support(self) -> list[str]:
        ...

    def retrieve(
        self,
        params: RetrieveParams,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[RetrieveResult]:
        ...

    def save(
        self,
        index_info: IndexInfo,
        *,
        embeddings_by_source_id: Mapping[str, Sequence[float]] | None = None,
    ) -> BatchSaveStats:
        ...

    def batch_save(
        self,
        index_info_list: Sequence[IndexInfo],
        *,
        embeddings_by_source_id: Mapping[str, Sequence[float]] | None = None,
    ) -> BatchSaveStats:
        ...

    def delete_by_knowledge_id_list(
        self,
        knowledge_id_list: Sequence[str],
        *,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        ...

    def estimate_storage_size(
        self,
        index_info_list: Sequence[IndexInfo],
        *,
        embeddings_by_source_id: Mapping[str, Sequence[float]] | None = None,
    ) -> int:
        ...

    def delete_by_chunk_id_list(
        self,
        chunk_id_list: Sequence[str],
        *,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        ...

    def delete_by_source_id_list(
        self,
        source_id_list: Sequence[str],
        *,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> int:
        ...

    def copy_indices(
        self,
        source_knowledge_base_id: str,
        source_to_target_kb_id_map: Mapping[str, str],
        source_to_target_chunk_id_map: Mapping[str, str],
        target_knowledge_base_id: str,
        *,
        dimension: int = 0,
        knowledge_type: str = "",
    ) -> None:
        ...

    def batch_update_chunk_enabled_status(self, chunk_status_map: Mapping[str, bool]) -> None:
        ...

    def batch_update_chunk_tag_id(self, chunk_tag_map: Mapping[str, str]) -> None:
        ...
