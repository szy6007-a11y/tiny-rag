from __future__ import annotations

from tiny_rag.agent import ChatModel

from .agent import AgentService, QueryService, config_with_scope
from .models import AgentQARequest, AgentQAResult, QueryRequest, QueryResult


class SessionService:
    def __init__(
        self,
        agent_service: AgentService,
        chat_model: ChatModel,
        *,
        default_system_prompt: str = "",
    ) -> None:
        self.agent_service = agent_service
        self.chat_model = chat_model
        self.default_system_prompt = default_system_prompt
        self.query_service = QueryService(agent_service)

    def agent_qa(self, request: AgentQARequest) -> AgentQAResult:
        config = config_with_scope(
            request.agent_config,
            knowledge_base_ids=request.knowledge_base_ids,
            knowledge_ids=request.knowledge_ids,
        )
        engine = self.agent_service.create_agent_engine(
            config,
            self.chat_model,
            tenant_id=request.tenant_id,
        )
        state = engine.execute(
            request.query,
            history=list(request.history or []),
            system_prompt=request.system_prompt or config.system_prompt or self.default_system_prompt,
        )
        return AgentQAResult(request=request, state=state)

    def knowledge_qa(self, request: QueryRequest) -> QueryResult:
        return self.query_service.retrieve(request)
