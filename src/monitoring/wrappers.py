"""Pipeline wrapper that records traces and metrics around RAG operations."""

from __future__ import annotations

import time
from typing import Any

from src.generation.citations import Citation
from src.monitoring.metrics import MetricsCollector
from src.monitoring.tracing import Tracer


class MonitoredRAGPipeline:
    """Wraps a :class:`~src.pipeline.RAGPipeline`, recording timing for
    ``ingest`` and ``query`` calls via a :class:`Tracer` and
    :class:`MetricsCollector`.

    All other attributes/methods are forwarded to the wrapped pipeline.
    """

    def __init__(self, pipeline: Any, tracer: Tracer, metrics: MetricsCollector) -> None:
        self._pipeline = pipeline
        self.tracer = tracer
        self.metrics = metrics

    def ingest(self, source: Any) -> int:
        """Time and record an ingestion call."""
        with self.tracer.span("ingest"):
            start = time.perf_counter()
            count = self._pipeline.ingest(source)
            self.metrics.record_timing("ingest_seconds", time.perf_counter() - start)
            self.metrics.increment("chunks_ingested", count)
            self.metrics.increment("ingest_calls")
        return count

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> tuple[str, list[Citation]]:
        """Time and record a query call."""
        with self.tracer.span("query"):
            start = time.perf_counter()
            answer, citations = self._pipeline.query(
                question,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
            )
            self.metrics.record_timing("query_seconds", time.perf_counter() - start)
            self.metrics.increment("query_calls")
            self.metrics.increment("citations_returned", len(citations))
        return answer, citations

    def __getattr__(self, name: str) -> Any:
        """Forward any other attribute access to the wrapped pipeline."""
        return getattr(self._pipeline, name)
