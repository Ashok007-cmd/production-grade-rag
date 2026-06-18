"""In-memory metrics collection with JSON export for CI summaries."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricsCollector:
    """Collects counters and timing samples for pipeline operations.

    Args:
        enabled: If False, all record methods are no-ops.
    """

    enabled: bool = True
    _counters: dict[str, int] = field(default_factory=dict)
    _timings: dict[str, list[float]] = field(default_factory=dict)

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        if not self.enabled:
            return
        self._counters[name] = self._counters.get(name, 0) + value

    def record_timing(self, name: str, seconds: float) -> None:
        """Record a timing sample (in seconds) under ``name``."""
        if not self.enabled:
            return
        self._timings.setdefault(name, []).append(seconds)

    def summary(self) -> dict[str, Any]:
        """Return aggregated counters and timing statistics."""
        timing_summary: dict[str, dict[str, float]] = {}
        for name, samples in self._timings.items():
            timing_summary[name] = {
                "count": len(samples),
                "total_s": round(sum(samples), 4),
                "avg_s": round(sum(samples) / len(samples), 4),
                "min_s": round(min(samples), 4),
                "max_s": round(max(samples), 4),
            }
        return {
            "counters": dict(self._counters),
            "timings": timing_summary,
        }

    def export_summary(self, path: str = "monitoring-summary.json") -> None:
        """Write the current summary to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        logger.info("Monitoring summary exported to %s", path)
