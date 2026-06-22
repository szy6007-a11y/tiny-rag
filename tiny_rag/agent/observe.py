from __future__ import annotations

import re
from dataclasses import dataclass

from .types import AgentStep, ChatResponse


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class ResponseVerdict:
    is_done: bool
    final_answer: str = ""
    empty_content: bool = False
    step: AgentStep | None = None


def strip_think_blocks(content: str) -> str:
    return THINK_BLOCK_RE.sub("", content or "").strip()


def analyze_response(
    response: ChatResponse,
    step: AgentStep,
) -> ResponseVerdict:
    if response.finish_reason == "content_filter" and len(response.tool_calls) == 0:
        answer = response.content
        if answer == "":
            answer = (
                "Sorry, this request was blocked by the content safety policy. "
                "Please try rephrasing your question."
            )
        return ResponseVerdict(is_done=True, final_answer=answer, step=step)

    if response.finish_reason == "stop" and len(response.tool_calls) == 0:
        answer = strip_think_blocks(response.content)
        return ResponseVerdict(
            is_done=True,
            final_answer=answer,
            empty_content=answer == "",
            step=step,
        )

    return ResponseVerdict(is_done=False, step=step)

