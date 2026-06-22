from __future__ import annotations

import httpx
from unittest import TestCase
from unittest.mock import patch

from tiny_rag.retrieval import (
    CHUNK_TYPE_FAQ,
    MATCH_TYPE_DIRECT_LOAD,
    RankResult,
    RerankConfig,
    RerankService,
    SiliconFlowReranker,
)
from tiny_rag.retrieval.models import SearchResult
from tiny_rag.retrieval.rerank import _rerank_endpoint


class StaticReranker:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def rerank(self, query, passages):
        self.calls.append((query, list(passages)))
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class FailingReranker:
    def rerank(self, query, passages):
        raise RuntimeError("rerank failed")


class CapturingSiliconFlowReranker(SiliconFlowReranker):
    def __init__(self, config, payload):
        super().__init__(config)
        self.payload = payload
        self.bodies = []

    def _post_with_retry(self, body, *, cancellation_token=None):
        self.bodies.append(dict(body))
        return httpx.Response(200, json=self.payload)


def result(chunk_id: str, *, score: float = 0.5, content: str | None = None, **kwargs):
    return SearchResult(
        id=chunk_id,
        content=content or f"content {chunk_id}",
        score=score,
        start_at=kwargs.pop("start_at", 0),
        end_at=kwargs.pop("end_at", 99),
        **kwargs,
    )


class RerankServiceTests(TestCase):
    def test_reranker_disabled_returns_original_results(self):
        items = [result("a"), result("b")]

        reranked = RerankService(None).rerank("q", items)

        self.assertEqual(reranked, items)

    def test_reranker_failure_falls_back_to_original_candidates(self):
        items = [result("a"), result("b")]
        service = RerankService(FailingReranker())

        reranked = service.rerank("q", items)

        self.assertEqual(reranked, [])
        self.assertTrue(service.last_api_error_fallback)
        self.assertEqual([item.id for item in service.last_fallback_results], ["a", "b"])

    def test_threshold_degrade_retries_when_first_pass_empty(self):
        reranker = StaticReranker(
            [
                [RankResult(index=0, relevance_score=0.1)],
                [RankResult(index=0, relevance_score=0.4)],
            ]
        )

        reranked = RerankService(reranker).rerank(
            "q",
            [result("a", score=0.2)],
            threshold=0.8,
        )

        self.assertEqual(len(reranker.calls), 2)
        self.assertEqual([item.id for item in reranked], ["a"])
        self.assertEqual(reranked[0].metadata["model_score"], "0.4000")

    def test_composite_score_uses_model_base_and_position_prior(self):
        reranker = StaticReranker([[RankResult(index=0, relevance_score=0.8)]])
        item = result("a", score=0.5, start_at=0, end_at=99)

        reranked = RerankService(reranker).rerank("q", [item], top_k=1, threshold=0.2)

        self.assertEqual([r.id for r in reranked], ["a"])
        self.assertAlmostEqual(reranked[0].score, (0.7 * 0.8 + 0.3 * 0.5) * 1.05)
        self.assertEqual(reranked[0].metadata["base_score"], "0.5000")

    def test_faq_boost_is_applied_after_composite_score(self):
        reranker = StaticReranker([[RankResult(index=0, relevance_score=0.6)]])
        item = result("faq", score=0.2, chunk_type=CHUNK_TYPE_FAQ)

        reranked = RerankService(reranker).rerank(
            "q",
            [item],
            top_k=1,
            faq_priority_enabled=True,
            faq_score_boost=2.0,
        )

        self.assertEqual(reranked[0].metadata["faq_boosted"], "true")
        self.assertEqual(reranked[0].score, 1.0)

    def test_mmr_removes_redundant_candidate(self):
        reranker = StaticReranker(
            [
                [
                    RankResult(index=0, relevance_score=0.9),
                    RankResult(index=1, relevance_score=0.89),
                    RankResult(index=2, relevance_score=0.8),
                ]
            ]
        )
        items = [
            result("a", score=0.0, content="alpha beta"),
            result("b", score=0.0, content="alpha beta"),
            result("c", score=0.0, content="gamma delta"),
        ]

        reranked = RerankService(reranker).rerank("q", items, top_k=2, threshold=0.2)

        self.assertEqual([item.id for item in reranked], ["a", "c"])

    def test_direct_load_bypasses_model_and_receives_composite_score(self):
        reranker = StaticReranker([[RankResult(index=0, relevance_score=0.5)]])
        direct = result("direct", score=0.5, match_type=MATCH_TYPE_DIRECT_LOAD)
        normal = result("normal", score=0.5)

        reranked = RerankService(reranker).rerank("q", [direct, normal], top_k=2)

        self.assertEqual({item.id for item in reranked}, {"direct", "normal"})
        self.assertEqual(direct.metadata["model_score"], "1.0000")


class SiliconFlowRerankerTests(TestCase):
    def test_rerank_config_env_uses_shared_siliconflow_key_and_defaults(self):
        with patch.dict(
            "os.environ",
            {
                "RERANK_PROVIDER": "",
                "RERANK_API_KEY": "",
                "SILICONFLOW_API_KEY": "test-key",
                "RERANK_BASE_URL": "",
                "RERANK_MODEL_NAME": "",
                "RERANK_TIMEOUT_SECONDS": "",
                "RERANK_MAX_RETRIES": "",
                "RERANK_MAX_CHUNKS_PER_DOC": "",
                "RERANK_OVERLAP_TOKENS": "",
            },
            clear=False,
        ):
            config = RerankConfig.from_env()

        self.assertEqual(config.provider, "siliconflow")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://api.siliconflow.cn/v1")
        self.assertEqual(config.model_name, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.max_chunks_per_doc, 1024)
        self.assertEqual(config.overlap_tokens, 80)

    def test_rerank_posts_siliconflow_payload_and_parses_results(self):
        reranker = CapturingSiliconFlowReranker(
            RerankConfig(
                api_key="test-key",
                model_name="BAAI/bge-reranker-v2-m3",
                max_chunks_per_doc=64,
                overlap_tokens=16,
            ),
            {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            },
        )

        ranked = reranker.rerank("latency", ["alpha", "beta"])

        self.assertEqual(
            reranker.bodies,
            [
                {
                    "model": "BAAI/bge-reranker-v2-m3",
                    "query": "latency",
                    "documents": ["alpha", "beta"],
                    "return_documents": False,
                    "top_n": 2,
                    "max_chunks_per_doc": 64,
                    "overlap_tokens": 16,
                }
            ],
        )
        self.assertEqual(ranked, [RankResult(index=1, relevance_score=0.91), RankResult(index=0, relevance_score=0.42)])

    def test_rerank_endpoint_accepts_base_url_or_full_path(self):
        self.assertEqual(
            _rerank_endpoint("https://api.siliconflow.cn/v1"),
            "https://api.siliconflow.cn/v1/rerank",
        )
        self.assertEqual(
            _rerank_endpoint("https://api.siliconflow.cn/v1/rerank"),
            "https://api.siliconflow.cn/v1/rerank",
        )
