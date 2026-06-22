from __future__ import annotations

import httpx
from unittest import TestCase
from unittest.mock import patch

from tiny_rag.embedding.models import EmbeddingConfig
from tiny_rag.embedding.openai import OpenAIEmbedder


class CapturingOpenAIEmbedder(OpenAIEmbedder):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.bodies = []

    def _post_with_retry(self, json_body, *, cancellation_token=None):
        self.bodies.append(dict(json_body))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            },
        )


class OpenAIEmbedderTests(TestCase):
    def test_siliconflow_provider_env_uses_shared_key_and_defaults(self):
        with patch.dict(
            "os.environ",
            {
                "EMBEDDING_PROVIDER": "siliconflow",
                "EMBEDDING_API_KEY": "",
                "SILICONFLOW_API_KEY": "test-key",
                "EMBEDDING_BASE_URL": "",
                "EMBEDDING_MODEL_NAME": "",
                "EMBEDDING_DIMENSIONS": "",
            },
            clear=False,
        ):
            config = EmbeddingConfig.from_env()

        self.assertEqual(config.provider, "siliconflow")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://api.siliconflow.cn/v1")
        self.assertEqual(config.model_name, "BAAI/bge-m3")
        self.assertEqual(config.dimensions, 1024)

    def test_batch_embed_omits_nonstandard_truncate_parameter_by_default(self):
        embedder = CapturingOpenAIEmbedder(
            api_key="test-key",
            base_url="https://api.siliconflow.com/v1",
            model_name="BAAI/bge-m3",
            dimensions=1024,
            supports_dimension_override=False,
        )

        embeddings = embedder.batch_embed(["alpha", "beta"])

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(
            embedder.bodies,
            [
                {
                    "model": "BAAI/bge-m3",
                    "input": ["alpha", "beta"],
                    "encoding_format": "float",
                }
            ],
        )

    def test_batch_embed_can_opt_into_truncate_and_dimensions_parameters(self):
        embedder = CapturingOpenAIEmbedder(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small",
            truncate_prompt_tokens=128,
            include_truncate_prompt_tokens=True,
            dimensions=256,
            supports_dimension_override=True,
        )

        embedder.batch_embed(["alpha"])

        self.assertEqual(
            embedder.bodies[0],
            {
                "model": "text-embedding-3-small",
                "input": ["alpha"],
                "encoding_format": "float",
                "truncate_prompt_tokens": 128,
                "dimensions": 256,
            },
        )
