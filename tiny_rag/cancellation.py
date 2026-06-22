from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class OperationCancelled(RuntimeError):
    pass


@dataclass
class CancellationToken:
    deadline: float | None = None
    _event: threading.Event = field(default_factory=threading.Event)
    reason: str = ""

    @classmethod
    def with_timeout(cls, timeout_seconds: float) -> "CancellationToken":
        if timeout_seconds <= 0:
            return cls()
        return cls(deadline=time.monotonic() + timeout_seconds)

    def cancel(self, reason: str = "operation cancelled") -> None:
        self.reason = reason
        self._event.set()

    def is_cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.cancel("operation deadline exceeded")
            return True
        return False

    def time_remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            message = self.reason or "operation cancelled"
            raise OperationCancelled(message)


def call_with_cancellation(
    func: Callable[..., Any],
    *args: Any,
    cancellation_token: CancellationToken | None = None,
    **kwargs: Any,
) -> Any:
    if cancellation_token is not None and accepts_cancellation_token(func):
        kwargs["cancellation_token"] = cancellation_token
    return func(*args, **kwargs)


def accepts_cancellation_token(func: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "cancellation_token":
            return True
    return False
