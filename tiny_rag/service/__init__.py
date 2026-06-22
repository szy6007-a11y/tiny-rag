"""Production-style orchestration services mirroring WeKnora's RAG/Agent flow."""

from .agent import AgentService, QueryService, config_with_scope, default_rerank_service_from_env
from .chat import ChatConfig, OpenAIChatModel, parse_chat_response
from .ingest import IngestService, default_parser_factory
from .models import (
    AgentQARequest,
    AgentQAResult,
    IngestRequest,
    IngestResult,
    KnowledgeBaseConfig,
    KnowledgeRecord,
    QueryRequest,
    QueryResult,
    default_knowledge_base_config,
)
from .session import SessionService

__all__ = [
    "AgentQARequest",
    "AgentQAResult",
    "AgentService",
    "ChatConfig",
    "IngestRequest",
    "IngestResult",
    "IngestService",
    "KnowledgeBaseConfig",
    "KnowledgeRecord",
    "OpenAIChatModel",
    "QueryRequest",
    "QueryResult",
    "QueryService",
    "SessionService",
    "config_with_scope",
    "default_knowledge_base_config",
    "default_parser_factory",
    "default_rerank_service_from_env",
    "parse_chat_response",
]
