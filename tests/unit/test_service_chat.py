from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from tiny_rag.service import ChatConfig


class ChatConfigTests(TestCase):
    def test_deepseek_provider_env_uses_deepseek_defaults(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "deepseek",
                "LLM_API_KEY": "",
                "DEEPSEEK_API_KEY": "test-key",
                "LLM_BASE_URL": "",
                "LLM_MODEL_NAME": "",
                "LLM_TIMEOUT_SECONDS": "",
                "LLM_MAX_RETRIES": "",
            },
            clear=False,
        ):
            config = ChatConfig.from_env()

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.model_name, "deepseek-v4-flash")
        self.assertEqual(config.timeout_seconds, 120.0)
        self.assertEqual(config.max_retries, 2)
