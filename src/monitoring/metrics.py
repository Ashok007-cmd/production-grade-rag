from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)


class MetricsCollector:
    """OpenTelemetry-based metrics collector for RAG pipeline observability.

    Records latency histograms (P50/P95), error counters, query counts,
    context usage, and cost estimates. Designed for Prometheus scraping.

    Usage:
        mc = MetricsCollector(enabled=settings.enabled, prefix="rag")
        mc.record_latency("retrieve", 0.342)
        mc.record_error("generate", "RateLimitError")
    """

    def __init__(self, enabled: bool = True, prefix: str = "rag") -> None:
        self.enabled = enabled
        self.prefix = prefix
        self._meter: Any = None
        self._latency_histograms: dict[str, Any] = {}
        self._error_counters: dict[str, Any] = {}
        self._query_counter: Any = None
        self._context_histogram: Any = None
        self._cost_histogram: Any = None
        self._token_histograms: dict[str, Any] = {}

        self.latencies: list[float] = []
        self.costs: list[float] = []
        self.errors_count: int = 0
        self.queries_count: int = 0

        if enabled:
            self._init_meter()

    def _init_meter(self) -> None:
        try:
            import logging

            logging.getLogger("opentelemetry").setLevel(logging.ERROR)
            logging.getLogger("opentelemetry.exporter.otlp.proto.http.metric_exporter").setLevel(
                logging.ERROR
            )

            from opentelemetry import metrics
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                PeriodicExportingMetricReader,
            )

            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/metrics",
                    timeout=settings.otel_export_timeout_ms // 1000,
                ),
                export_interval_millis=settings.otel_export_interval_ms,
                export_timeout_millis=settings.otel_export_timeout_ms,
            )

            provider = MeterProvider(metric_readers=[reader])
            metrics.set_meter_provider(provider)

            import atexit

            atexit.register(provider.shutdown)

            self._meter = metrics.get_meter(settings.otel_service_name)

            for step in ["retrieve", "rerank", "generate", "query_total"]:
                self._latency_histograms[step] = self._meter.create_histogram(
                    name=f"{self.prefix}_{step}_latency_seconds",
                    description=f"Latency of {step} step in seconds",
                    unit="s",
                )

            for step in ["retrieve", "rerank", "generate", "query_total"]:
                self._error_counters[step] = self._meter.create_counter(
                    name=f"{self.prefix}_{step}_errors_total",
                    description=f"Total errors in {step} step",
                    unit="1",
                )

            self._query_counter = self._meter.create_counter(
                name=f"{self.prefix}_queries_total",
                description="Total number of RAG queries",
                unit="1",
            )

            self._context_histogram = self._meter.create_histogram(
                name=f"{self.prefix}_context_chunks_per_query",
                description="Number of context chunks used per query",
                unit="1",
            )

            self._cost_histogram = self._meter.create_histogram(
                name=f"{self.prefix}_cost_per_query_dollars",
                description="Estimated cost per query in USD",
                unit="$",
            )

            for token_type in ["prompt", "completion", "total"]:
                self._token_histograms[token_type] = self._meter.create_histogram(
                    name=f"{self.prefix}_tokens_{token_type}",
                    description=f"Number of {token_type} tokens per query",
                    unit="1",
                )

            logger.info("OTel meter initialised with prefix '%s'", self.prefix)

        except ImportError as exc:
            logger.warning("OTel packages not installed, metrics disabled: %s", exc)
            self.enabled = False

    def record_latency(self, step: str, seconds: float) -> None:
        if step == "query_total" or step == "query":
            self.latencies.append(seconds)
        if not self.enabled or self._meter is None:
            return
        if step == "query":
            step = "query_total"
        hist = self._latency_histograms.get(step)
        if hist:
            hist.record(seconds)

    def record_error(self, step: str, error_type: str) -> None:
        self.errors_count += 1
        if not self.enabled or self._meter is None:
            return
        if step == "query":
            step = "query_total"
        counter = self._error_counters.get(step)
        if counter:
            counter.add(1, {"error_type": error_type})

    def record_query_count(self) -> None:
        self.queries_count += 1
        if not self.enabled or self._meter is None:
            return
        self._query_counter.add(1)

    def record_context_count(self, count: int) -> None:
        if not self.enabled or self._context_histogram is None:
            return
        self._context_histogram.record(count)

    def record_cost(self, cost: float) -> None:
        self.costs.append(cost)
        if not self.enabled or self._cost_histogram is None:
            return
        self._cost_histogram.record(cost)

    def record_tokens(self, prompt: int, completion: int, total: int) -> None:
        if not self.enabled:
            return
        mapping = {"prompt": prompt, "completion": completion, "total": total}
        for token_type, value in mapping.items():
            hist = self._token_histograms.get(token_type)
            if hist:
                hist.record(value)

    def export_summary(self, path: str | Path) -> None:
        import json

        if not self.latencies:
            p50 = 0.0
            p95 = 0.0
        else:
            sorted_lats = sorted(self.latencies)
            n = len(sorted_lats)
            p50 = sorted_lats[min(int(n * 0.50), n - 1)]
            p95 = sorted_lats[min(int(n * 0.95), n - 1)]

        avg_cost = sum(self.costs) / len(self.costs) if self.costs else 0.0

        summary = {
            "p50_latency": p50,
            "p95_latency": p95,
            "avg_cost": avg_cost,
            "total_queries": self.queries_count,
            "errors": self.errors_count,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import os

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(path, flags, 0o600)
        with open(fd, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Metrics summary exported to %s", path)
