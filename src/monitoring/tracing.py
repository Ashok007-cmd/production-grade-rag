"""Minimal in-process span tracer.

Not a replacement for OpenTelemetry — a dependency-free way to record
nested timing spans for local runs and CI evaluation summaries.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single recorded timing span."""

    name: str
    duration_s: float
    parent: str | None = None


@dataclass
class Tracer:
    """Records nested timing spans via :meth:`span`.

    Args:
        enabled: If False, :meth:`span` is a no-op (zero overhead).
    """

    enabled: bool = True
    spans: list[Span] = field(default_factory=list)
    _stack: list[str] = field(default_factory=list)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Context manager that records the wall-clock duration of a block."""
        if not self.enabled:
            yield
            return

        parent = self._stack[-1] if self._stack else None
        self._stack.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._stack.pop()
            self.spans.append(Span(name=name, duration_s=duration, parent=parent))
            logger.debug("Span '%s' took %.4fs (parent=%s)", name, duration, parent)

    def summary(self) -> dict[str, dict[str, float]]:
        """Aggregate recorded spans by name: count, total, and average duration."""
        agg: dict[str, dict[str, float]] = {}
        for s in self.spans:
            entry = agg.setdefault(s.name, {"count": 0, "total_s": 0.0})
            entry["count"] += 1
            entry["total_s"] += s.duration_s
        for entry in agg.values():
            entry["avg_s"] = entry["total_s"] / entry["count"]
        return agg
