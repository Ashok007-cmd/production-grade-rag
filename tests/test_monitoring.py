"""Tests for src.monitoring (tracing, metrics, pipeline wrapper)."""

from __future__ import annotations

import json

from src.generation.citations import Citation
from src.monitoring.metrics import MetricsCollector
from src.monitoring.tracing import Tracer
from src.monitoring.wrappers import MonitoredRAGPipeline


class TestTracer:
    def test_span_records_duration(self) -> None:
        tracer = Tracer(enabled=True)
        with tracer.span("op"):
            pass
        assert len(tracer.spans) == 1
        assert tracer.spans[0].name == "op"
        assert tracer.spans[0].duration_s >= 0

    def test_disabled_tracer_records_nothing(self) -> None:
        tracer = Tracer(enabled=False)
        with tracer.span("op"):
            pass
        assert tracer.spans == []

    def test_summary_aggregates_by_name(self) -> None:
        tracer = Tracer(enabled=True)
        with tracer.span("op"):
            pass
        with tracer.span("op"):
            pass
        summary = tracer.summary()
        assert summary["op"]["count"] == 2
        assert "avg_s" in summary["op"]


class TestMetricsCollector:
    def test_increment_and_timing(self) -> None:
        metrics = MetricsCollector(enabled=True)
        metrics.increment("calls")
        metrics.increment("calls")
        metrics.record_timing("latency", 0.5)

        summary = metrics.summary()
        assert summary["counters"]["calls"] == 2
        assert summary["timings"]["latency"]["count"] == 1
        assert summary["timings"]["latency"]["avg_s"] == 0.5

    def test_disabled_collector_is_noop(self) -> None:
        metrics = MetricsCollector(enabled=False)
        metrics.increment("calls")
        metrics.record_timing("latency", 0.5)
        summary = metrics.summary()
        assert summary["counters"] == {}
        assert summary["timings"] == {}

    def test_export_summary_writes_json(self, tmp_path) -> None:
        metrics = MetricsCollector(enabled=True)
        metrics.increment("calls")
        out = tmp_path / "summary.json"
        metrics.export_summary(str(out))

        data = json.loads(out.read_text())
        assert data["counters"]["calls"] == 1


class _FakePipeline:
    def __init__(self) -> None:
        self.queried_with: tuple | None = None

    def ingest(self, source) -> int:
        return 3

    def query(self, question, top_k=None, use_hybrid=False, use_reranker=False):
        self.queried_with = (question, top_k, use_hybrid, use_reranker)
        return "answer", [Citation(chunk_id="c1", source="s", filename="f", text_snippet="t")]

    def stats(self) -> dict:
        return {"chunks_in_store": 3}


class TestMonitoredRAGPipeline:
    def test_ingest_records_metrics(self) -> None:
        tracer = Tracer(enabled=True)
        metrics = MetricsCollector(enabled=True)
        wrapped = MonitoredRAGPipeline(_FakePipeline(), tracer=tracer, metrics=metrics)

        count = wrapped.ingest("some/path")

        assert count == 3
        assert metrics.summary()["counters"]["chunks_ingested"] == 3
        assert metrics.summary()["counters"]["ingest_calls"] == 1
        assert "ingest_seconds" in metrics.summary()["timings"]

    def test_query_records_metrics(self) -> None:
        tracer = Tracer(enabled=True)
        metrics = MetricsCollector(enabled=True)
        wrapped = MonitoredRAGPipeline(_FakePipeline(), tracer=tracer, metrics=metrics)

        answer, citations = wrapped.query("What is RAG?", use_hybrid=True)

        assert answer == "answer"
        assert len(citations) == 1
        assert metrics.summary()["counters"]["query_calls"] == 1
        assert metrics.summary()["counters"]["citations_returned"] == 1
        assert "query_seconds" in metrics.summary()["timings"]

    def test_forwards_unknown_attributes(self) -> None:
        tracer = Tracer(enabled=True)
        metrics = MetricsCollector(enabled=True)
        wrapped = MonitoredRAGPipeline(_FakePipeline(), tracer=tracer, metrics=metrics)

        assert wrapped.stats() == {"chunks_in_store": 3}
