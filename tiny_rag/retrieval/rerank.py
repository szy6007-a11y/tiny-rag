from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

from tiny_rag.cancellation import CancellationToken, OperationCancelled, call_with_cancellation

from .models import (
    CHUNK_TYPE_FAQ,
    MATCH_TYPE_DIRECT_LOAD,
    RankResult,
    Reranker,
    SearchResult,
)
from .utils import clean_passage_for_rerank, clamp_float, jaccard, tokenize_simple


RESERVED_HEADER_NAMES = {"authorization", "content-type"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass(frozen=True)
class RerankConfig:
    provider: str = "siliconflow"
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    max_chunks_per_doc: int = 1024
    overlap_tokens: int = 80
    return_documents: bool = False
    custom_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "RerankConfig":
        return cls(
            provider=_env_str("RERANK_PROVIDER", "siliconflow"),
            api_key=_env_str("RERANK_API_KEY", os.getenv("SILICONFLOW_API_KEY", "")),
            base_url=_env_str("RERANK_BASE_URL", "https://api.siliconflow.cn/v1"),
            model_name=_env_str("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
            timeout_seconds=_env_float("RERANK_TIMEOUT_SECONDS", 60.0),
            max_retries=_env_int("RERANK_MAX_RETRIES", 2),
            max_chunks_per_doc=_env_int("RERANK_MAX_CHUNKS_PER_DOC", 1024),
            overlap_tokens=_env_int("RERANK_OVERLAP_TOKENS", 80),
            return_documents=os.getenv("RERANK_RETURN_DOCUMENTS", "").strip().lower()
            in {"1", "true", "yes", "y", "on"},
        )


@dataclass
class SiliconFlowReranker:
    config: RerankConfig

    @classmethod
    def from_env(cls) -> "SiliconFlowReranker":
        return cls(RerankConfig.from_env())

    def __post_init__(self) -> None:
        if self.config.provider.lower() != "siliconflow":
            raise ValueError(f"unsupported rerank provider: {self.config.provider}")
        if not self.config.model_name:
            raise ValueError("rerank model name is required")

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[RankResult]:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        if not passages:
            return []
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "query": query,
            "documents": list(passages),
            "return_documents": self.config.return_documents,
            "top_n": len(passages),
            "max_chunks_per_doc": self.config.max_chunks_per_doc,
            "overlap_tokens": self.config.overlap_tokens,
        }
        response = self._post_with_retry(body, cancellation_token=cancellation_token)
        if response.status_code != 200:
            body_text = response.text
            if len(body_text) > 1000:
                body_text = body_text[:1000] + "... (truncated)"
            raise RuntimeError(
                f"Rerank API error: Http Status {response.status_code}, Response: {body_text}"
            )
        payload = response.json()
        results = payload.get("results") or []
        return [_coerce_rank_result(item) for item in results]

    def get_model_name(self) -> str:
        return self.config.model_name

    def _post_with_retry(
        self,
        body: dict[str, Any],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        url = _rerank_endpoint(self.config.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.config.api_key,
        }
        for key, value in self.config.custom_headers.items():
            if key.lower() not in RESERVED_HEADER_NAMES:
                headers[key] = value

        timeout = self.config.timeout_seconds
        if cancellation_token is not None and cancellation_token.time_remaining() is not None:
            timeout = min(timeout, max(0.001, cancellation_token.time_remaining() or 0.001))

        with httpx.Client(timeout=timeout) as client:
            for attempt in range(self.config.max_retries + 1):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                if attempt > 0:
                    _sleep_with_cancellation(
                        min(2 ** (attempt - 1), 10),
                        cancellation_token=cancellation_token,
                    )
                try:
                    return client.post(url, headers=headers, json=body)
                except Exception as exc:
                    last_error = exc

        if last_error is not None:
            raise RuntimeError(f"send rerank request: {last_error}") from last_error
        raise RuntimeError("send rerank request failed")


def _rerank_endpoint(base_url: str) -> str:
    trimmed = (base_url or "https://api.siliconflow.cn/v1").rstrip("/")
    if trimmed.endswith("/rerank"):
        return trimmed
    return trimmed + "/rerank"


def _sleep_with_cancellation(
    seconds: float,
    *,
    cancellation_token: CancellationToken | None = None,
) -> None:
    if cancellation_token is None:
        time.sleep(seconds)
        return
    deadline = time.monotonic() + seconds
    while True:
        cancellation_token.raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


class RerankService:
    def __init__(self, reranker: Reranker | None = None) -> None:
        self.reranker = reranker
        self.last_api_error_fallback = False
        self.last_fallback_results: list[SearchResult] = []

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        *,
        top_k: int = 10,
        threshold: float = 0.2,
        faq_priority_enabled: bool = False,
        faq_score_boost: float = 1.0,
        cancellation_token: CancellationToken | None = None,
    ) -> list[SearchResult]:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        self.last_api_error_fallback = False
        self.last_fallback_results = []

        if not results:
            return []
        if self.reranker is None:
            return list(results)

        passages: list[str] = []
        candidates: list[SearchResult] = []
        direct_load_results: list[SearchResult] = []

        for result in results:
            if result.match_type == MATCH_TYPE_DIRECT_LOAD:
                direct_load_results.append(result)
                continue
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            passage = get_enriched_passage(result)
            if passage.strip() == "":
                continue
            passages.append(passage)
            candidates.append(result)

        rerank_resp: list[RankResult] = []
        if candidates:
            original_threshold = threshold
            try:
                rerank_resp = self._rerank(
                    query,
                    passages,
                    candidates,
                    threshold,
                    cancellation_token=cancellation_token,
                )
            except OperationCancelled:
                raise
            except Exception:
                self.last_api_error_fallback = True
                self.last_fallback_results = direct_load_results + candidates
                return []

            if len(rerank_resp) == 0 and original_threshold > 0.3:
                degraded_threshold = original_threshold * 0.7
                if degraded_threshold < 0.3:
                    degraded_threshold = 0.3
                try:
                    rerank_resp = self._rerank(
                        query,
                        passages,
                        candidates,
                        degraded_threshold,
                        cancellation_token=cancellation_token,
                    )
                except OperationCancelled:
                    raise
                except Exception:
                    self.last_api_error_fallback = True
                    self.last_fallback_results = direct_load_results + candidates
                    return []

        for result in results:
            result.metadata = result.metadata or {}

        reranked: list[SearchResult] = []
        for rank in rerank_resp:
            if rank.index >= len(candidates) or rank.index < 0:
                continue
            sr = candidates[rank.index]
            base = sr.score
            model_score = rank.relevance_score
            sr.metadata["base_score"] = f"{base:.4f}"
            sr.metadata["model_score"] = f"{model_score:.4f}"
            sr.score = composite_score(sr, model_score, base)
            if faq_priority_enabled and faq_score_boost > 1.0 and sr.chunk_type == CHUNK_TYPE_FAQ:
                original_score = sr.score
                sr.score = min(sr.score * faq_score_boost, 1.0)
                sr.metadata["faq_boosted"] = "true"
                sr.metadata["faq_original_score"] = f"{original_score:.4f}"
            reranked.append(sr)

        for sr in direct_load_results:
            base = sr.score
            model_score = 1.0
            sr.metadata["base_score"] = f"{base:.4f}"
            sr.metadata["model_score"] = f"{model_score:.4f}"
            sr.score = composite_score(sr, model_score, base)
            reranked.append(sr)

        return apply_mmr(reranked, min(len(reranked), max(1, top_k)), 0.7)

    def _rerank(
        self,
        query: str,
        passages: Sequence[str],
        candidates: Sequence[SearchResult],
        threshold: float,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[RankResult]:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        clean_passages: list[str] = []
        clean_candidates: list[SearchResult] = []
        for index, passage in enumerate(passages):
            if passage.strip() == "":
                continue
            clean_passages.append(passage)
            if index < len(candidates):
                clean_candidates.append(candidates[index])
        if not clean_passages:
            return []

        response = _call_reranker(
            self.reranker,
            query,
            clean_passages,
            cancellation_token=cancellation_token,
        )
        rank_filter: list[RankResult] = []
        for result in response:
            if result.index >= len(clean_candidates) or result.index < 0:
                continue
            if result.relevance_score >= threshold:
                rank_filter.append(result)

        fallback_min_score = 0.15
        if not rank_filter and response and response[0].relevance_score >= fallback_min_score:
            rank_filter = response[:1]
        return rank_filter


def _call_reranker(
    reranker: Reranker | None,
    query: str,
    passages: Sequence[str],
    *,
    cancellation_token: CancellationToken | None = None,
) -> list[RankResult]:
    if reranker is None:
        return []
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    if hasattr(reranker, "rerank"):
        raw = call_with_cancellation(
            reranker.rerank,
            query,
            passages,
            cancellation_token=cancellation_token,
        )
    else:
        raw = call_with_cancellation(
            reranker.Rerank,
            query,
            passages,
            cancellation_token=cancellation_token,
        )
    return [_coerce_rank_result(item) for item in raw]


def _coerce_rank_result(item: Any) -> RankResult:
    if isinstance(item, RankResult):
        return item
    if isinstance(item, dict):
        return RankResult(
            index=int(item.get("index", item.get("Index", 0))),
            relevance_score=float(
                item.get("relevance_score", item.get("RelevanceScore", item.get("score", 0.0)))
            ),
        )
    index = getattr(item, "index", getattr(item, "Index", None))
    score = getattr(
        item,
        "relevance_score",
        getattr(item, "RelevanceScore", getattr(item, "score", None)),
    )
    if index is not None and score is not None:
        return RankResult(index=int(index), relevance_score=float(score))
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return RankResult(index=int(item[0]), relevance_score=float(item[1]))
    raise TypeError(f"unsupported rank result: {item!r}")


def composite_score(sr: SearchResult, model_score: float, base_score: float) -> float:
    position_prior = 1.0
    if sr.start_at >= 0:
        position_prior += clamp_float(
            1.0 - float(sr.start_at) / float(sr.end_at + 1),
            -0.05,
            0.05,
        )
    composite = 0.7 * model_score + 0.3 * base_score
    composite *= position_prior
    if composite < 0:
        return 0.0
    if composite > 1:
        return 1.0
    return composite


def apply_mmr(results: Sequence[SearchResult], k: int, lambda_value: float) -> list[SearchResult]:
    if k <= 0 or not results:
        return []

    all_token_sets = [tokenize_simple(get_enriched_passage(result)) for result in results]
    selected: list[SearchResult] = []
    selected_token_sets: list[set[str]] = []
    selected_indices: set[int] = set()

    while len(selected) < k and len(selected_indices) < len(results):
        best_index = -1
        best_score = -1.0
        for index, result in enumerate(results):
            if index in selected_indices:
                continue
            relevance = result.score
            redundancy = 0.0
            for selected_tokens in selected_token_sets:
                sim = jaccard(all_token_sets[index], selected_tokens)
                if sim > redundancy:
                    redundancy = sim
            mmr = lambda_value * relevance - (1.0 - lambda_value) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_index = index
        if best_index < 0:
            break
        selected.append(results[best_index])
        selected_token_sets.append(all_token_sets[best_index])
        selected_indices.add(best_index)
    return selected


def get_enriched_passage(result: SearchResult) -> str:
    combined_text = clean_passage_for_rerank(result.content)
    enrichments: list[str] = []

    if result.image_info:
        try:
            image_infos = json.loads(result.image_info)
        except (TypeError, ValueError):
            image_infos = []
        if isinstance(image_infos, list):
            for image in image_infos:
                if not isinstance(image, dict):
                    continue
                caption = image.get("caption") or image.get("Caption")
                ocr_text = image.get("ocr_text") or image.get("OCRText")
                if caption:
                    enrichments.append(str(caption))
                if ocr_text:
                    enrichments.append(str(ocr_text))

    metadata = _load_chunk_metadata(result.chunk_metadata)
    generated_questions = metadata.get("generated_questions") if isinstance(metadata, dict) else None
    if isinstance(generated_questions, list):
        questions: list[str] = []
        for question in generated_questions:
            if isinstance(question, dict):
                value = question.get("question") or question.get("Question")
                if value:
                    questions.append(str(value))
            elif isinstance(question, str):
                questions.append(question)
        if questions:
            enrichments.append("; ".join(questions))

    if not enrichments:
        return combined_text
    if combined_text:
        combined_text += "\n\n"
    combined_text += "\n".join(enrichments)
    return combined_text


def _load_chunk_metadata(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}
