"""Token usage and latency tracking context for RAG API requests."""

from __future__ import annotations

import contextvars

# ContextVar to accumulate token counts and latency for the current request context.
# Highly compatible with async FastAPI execution.
request_usage: contextvars.ContextVar[UsageTracker | None] = contextvars.ContextVar(
    "request_usage", default=None
)


class UsageTracker:
    """Tracks token consumption and API latency for downstream headers / metrics."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_latency: float = 0.0

    def add_call(self, prompt_tokens: int, completion_tokens: int, latency: float) -> None:
        """Accumulate token counts and latency from an LLM call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_latency += latency
