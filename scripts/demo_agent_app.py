#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_rag.agent import AgentConfig, ChatOptions, ChatResponse, FunctionCall, LLMToolCall, Message
from tiny_rag.agent.tools import ToolKnowledgeSearch
from tiny_rag.chunking import SplitterConfig
from tiny_rag.persistence import connect
from tiny_rag.retrieval import RankResult, RerankService
from tiny_rag.service import (
    AgentQARequest,
    AgentService,
    IngestRequest,
    IngestService,
    KnowledgeBaseConfig,
    OpenAIChatModel,
    SessionService,
)


DEFAULT_DOCUMENT = """# Operations Runbook

## Retrieval Service

The retrieval service has a latency_budget of 120ms for interactive knowledge
lookups. The index path is sqlite-vector, with keyword fallback enabled for
literal terms and identifiers.

## Incident Review

Every production incident needs an incident review within 24 hours. The owner is
the on-call incident commander. The review should include customer impact,
retrieval traces, failed tool calls, and the rollback decision.

## Release Checklist

Before release, operators verify prompt compatibility, run a smoke query against
the knowledge base, and confirm that cited chunks come from the current index.
"""

DEFAULT_QUERY = "Who owns the incident review and how long is the SLA?"

DEFAULT_SYSTEM_PROMPT = """You are a RAG agent for an operations knowledge base.
Use the available tools before answering questions about the indexed document.
Answer only from retrieved knowledge and keep the final answer concise."""


class DemoEmbedder:
    def __init__(self, dimensions: int = 6) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        return self._vector(text)

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def get_model_name(self) -> str:
        return "demo-deterministic-embedding"

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_model_id(self) -> str:
        return self.get_model_name()

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        features = [
            1.0 if any(term in lowered for term in ("latency", "budget", "120ms")) else 0.1,
            1.0 if any(term in lowered for term in ("incident", "review", "sla", "24 hour")) else 0.1,
            1.0 if any(term in lowered for term in ("owner", "commander", "on-call")) else 0.1,
            1.0 if any(term in lowered for term in ("release", "checklist", "smoke")) else 0.1,
            1.0 if any(term in lowered for term in ("tool", "trace", "rollback")) else 0.1,
            float(len(text) % 101 + 1) / 101.0,
        ]
        return features[: self.dimensions]


class DemoReranker:
    def rerank(self, query: str, passages, **kwargs) -> list[RankResult]:
        del query, kwargs
        return [RankResult(index=index, relevance_score=1.0) for index, _ in enumerate(passages)]


class ScriptedDemoChatModel:
    def __init__(self, query: str, knowledge_base_id: str) -> None:
        self.query = query
        self.knowledge_base_id = knowledge_base_id
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], opts: ChatOptions) -> ChatResponse:
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            del opts
            args = {
                "queries": [self.query],
                "knowledge_base_ids": [self.knowledge_base_id],
            }
            return ChatResponse(
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="demo-knowledge-search",
                        function=FunctionCall(
                            name=ToolKnowledgeSearch,
                            arguments=json.dumps(args, ensure_ascii=False),
                        ),
                    )
                ],
            )

        tool_message = next(
            (message for message in reversed(messages) if message.role == "tool"),
            Message(role="tool", content=""),
        )
        return ChatResponse(content=build_demo_answer(tool_message.content), finish_reason="stop")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end Tiny RAG Agent demo in memory."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--document", help="Optional UTF-8 Markdown/text document to ingest")
    parser.add_argument("--live", action="store_true", help="Use OpenAI-compatible LLM env vars")
    parser.add_argument("--chunk-size", type=int, default=220)
    args = parser.parse_args(argv)

    document_text, file_name = load_document(args.document)
    query = args.query.strip() or DEFAULT_QUERY

    conn = connect(":memory:")
    try:
        result = run_demo(
            conn=conn,
            document_text=document_text,
            file_name=file_name,
            query=query,
            use_live_model=args.live,
            chunk_size=args.chunk_size,
        )
        print_demo_result(result)
    finally:
        conn.close()
    return 0


def load_document(path: str | None) -> tuple[str, str]:
    if path:
        doc_path = Path(path)
        return doc_path.read_text(encoding="utf-8"), doc_path.name
    return DEFAULT_DOCUMENT, "operations-runbook.md"


def run_demo(
    *,
    conn,
    document_text: str,
    file_name: str,
    query: str,
    use_live_model: bool,
    chunk_size: int,
) -> dict[str, object]:
    tenant_id = 1
    knowledge_base_id = "demo-ops-kb"
    knowledge_id = "demo-runbook"
    embedder = DemoEmbedder()
    kb = KnowledgeBaseConfig(
        id=knowledge_base_id,
        tenant_id=tenant_id,
        name="Demo Operations KB",
        embedding_model_id=embedder.get_model_id(),
        keyword_enabled=True,
        vector_enabled=True,
    )

    ingest_service = IngestService(conn, embedder=embedder, knowledge_bases={kb.id: kb})
    ingest_result = ingest_service.ingest_document(
        IngestRequest(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            title=Path(file_name).stem or "Operations Runbook",
            file_name=file_name,
            file_type=Path(file_name).suffix.lstrip(".") or "md",
            content=document_text,
            chunk_config=SplitterConfig(
                chunk_size=chunk_size,
                chunk_overlap=40,
                separators=["\n\n", "\n", ". "],
                strategy="auto",
            ),
            knowledge_base=kb,
        )
    )

    agent_service = AgentService(
        conn,
        repository=ingest_service.repository,
        chunk_repository=ingest_service.chunk_repository,
        embedder=embedder,
        knowledge_bases=ingest_service.knowledge_bases,
        knowledge_records=ingest_service.knowledge_records,
        rerank_service=RerankService(DemoReranker()),
    )
    chat_model = (
        OpenAIChatModel.from_env()
        if use_live_model
        else ScriptedDemoChatModel(query, knowledge_base_id)
    )
    session_service = SessionService(agent_service, chat_model, default_system_prompt=DEFAULT_SYSTEM_PROMPT)
    agent_result = session_service.agent_qa(
        AgentQARequest(
            query=query,
            tenant_id=tenant_id,
            knowledge_base_ids=[knowledge_base_id],
            agent_config=AgentConfig(
                max_iterations=4,
                temperature=0.1,
                max_tool_output_chars=8000,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            ),
        )
    )
    return {
        "file_name": file_name,
        "query": query,
        "ingest": ingest_result,
        "agent": agent_result,
        "live": use_live_model,
    }


def print_demo_result(result: dict[str, object]) -> None:
    ingest_result = result["ingest"]
    agent_result = result["agent"]

    print("Tiny RAG Agent demo")
    print("===================")
    print(f"document: {result['file_name']}")
    print(f"query: {result['query']}")
    print()
    print("1. ingest -> chunk -> index")
    print(f"   chunks: {ingest_result.persist_result.text_count}")
    print(f"   keyword rows: {ingest_result.index_stats.fts_count}")
    print(f"   vector rows: {ingest_result.index_stats.vector_count}")
    print()
    print("2. agent tool trace")
    for step in agent_result.state.round_steps:
        if not step.tool_calls:
            continue
        print(f"   round {step.iteration}:")
        for call in step.tool_calls:
            status = "ok" if call.result.success else "error"
            print(f"   - {call.name}({json.dumps(call.args, ensure_ascii=False)}) -> {status}")
            if call.result.output:
                print(f"     preview: {single_line_preview(call.result.output)}")
    print()
    print("3. final answer")
    print(indent(agent_result.state.final_answer.strip() or "(empty)"))


def build_demo_answer(tool_content: str) -> str:
    if "No relevant content found" in tool_content:
        return "I could not find relevant information in the indexed knowledge base."

    content = extract_first_content(tool_content)
    lowered = content.lower()
    if "incident review" in lowered and "24 hours" in lowered and "commander" in lowered:
        return (
            "The incident review is owned by the on-call incident commander, "
            "and the review SLA is within 24 hours."
        )
    if content:
        return "The retrieved knowledge says:\n\n" + content[:700].strip()
    return "The tool returned results, but I could not extract a readable content snippet."


def extract_first_content(tool_content: str) -> str:
    match = re.search(r"<content>(.*?)</content>", tool_content, flags=re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def single_line_preview(text: str, limit: int = 220) -> str:
    preview = re.sub(r"\s+", " ", text).strip()
    if len(preview) <= limit:
        return preview
    return preview[:limit] + "..."


def indent(text: str) -> str:
    return "\n".join("   " + line for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
