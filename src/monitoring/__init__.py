"""Lightweight observability for the RAG pipeline (tracing + metrics).

Disabled by default; enable by setting the ``MONITOR_ENABLED=true`` environment
variable, which causes ``scripts/evaluate.py`` to wrap the pipeline with
:class:`~src.monitoring.wrappers.MonitoredRAGPipeline`.
"""

from __future__ import annotations

from src.monitoring.metrics import MetricsCollector
from src.monitoring.tracing import Tracer
from src.monitoring.wrappers import MonitoredRAGPipeline

__all__ = ["MetricsCollector", "Tracer", "MonitoredRAGPipeline"]
