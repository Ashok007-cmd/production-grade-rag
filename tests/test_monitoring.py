"""Tests for src.monitoring (tracing, metrics, pipeline wrapper)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.monitoring.extensions import GuardrailExtension, OTelMetricsExtension
from src.monitoring.metrics import MetricsCollector
from src.monitoring.tracing import Tracer
from src.monitoring.wrappers import MonitoredRAGPipeline


class TestTracer:
    def test_trace_step_disabled_yields_dict_span(self) -> None:
        """When Langfuse is unavailable, trace_step yields a plain dict span."""
        tracer = Tracer(enabled=True)  # no langfuse key → _langfuse stays None
        with tracer.trace_step("retrieve", input={"query": "q"}) as span:
            assert span["name"] == "retrieve"
            span["output"] = {"count": 5}
        # After context exit the span is cleaned up; no exception means it worked.

    def test_trace_step_disabled_mode(self) -> None:
        tracer = Tracer(enabled=False)
        with tracer.trace_step("retrieve") as span:
            span["output"] = {}  # should not raise

    def test_get_trace_id_is_uuid_string(self) -> None:
        tracer = Tracer(enabled=True)
        tid = tracer.get_trace_id()
        assert isinstance(tid, str) and len(tid) == 36  # UUID4 format

    def test_nested_spans_cleaned_up(self) -> None:
        tracer = Tracer(enabled=True)
        with tracer.trace_step("outer") as outer:
            with tracer.trace_step("inner") as inner:
                inner["output"] = {}
            outer["output"] = {}
        # After both exits, active_spans list should be empty per thread-local.
        assert tracer._active_spans == []

    def test_trace_step_resets_trace_id_after_root_exits(self) -> None:
        tracer = Tracer(enabled=True)
        with tracer.trace_step("root") as _:
            pass
        # After root span exits with disabled Langfuse, trace_id resets.
        assert tracer._trace_id == ""


class TestMetricsCollector:
    def test_record_latency_appends_to_list(self) -> None:
        mc = MetricsCollector(enabled=False)
        mc.record_latency("query_total", 0.42)
        assert 0.42 in mc.latencies

    def test_record_error_increments_count(self) -> None:
        mc = MetricsCollector(enabled=False)
        mc.record_error("generate", "RateLimitError")
        assert mc.errors_count == 1

    def test_record_query_count_increments(self) -> None:
        mc = MetricsCollector(enabled=False)
        mc.record_query_count()
        mc.record_query_count()
        assert mc.queries_count == 2

    def test_record_cost_appends(self) -> None:
        mc = MetricsCollector(enabled=False)
        mc.record_cost(0.001)
        assert 0.001 in mc.costs

    def test_export_summary_writes_json(self, tmp_path) -> None:
        mc = MetricsCollector(enabled=False)
        mc.record_latency("query_total", 0.1)
        mc.record_latency("query_total", 0.3)
        mc.record_query_count()
        mc.record_cost(0.005)

        out = tmp_path / "metrics.json"
        mc.export_summary(out)

        data = json.loads(out.read_text())
        assert data["total_queries"] == 1
        assert data["p50_latency"] >= 0
        assert data["avg_cost"] == 0.005

    def test_disabled_skips_otel_init(self) -> None:
        mc = MetricsCollector(enabled=False)
        assert mc._meter is None


class TestGuardrailExtension:
    def test_blocks_query_with_keyword(self) -> None:
        guard = GuardrailExtension(blocked_keywords=["forbidden"])
        import pytest

        with pytest.raises(ValueError, match="forbidden"):
            guard.on_query_start("tell me forbidden secrets", {})

    def test_allows_clean_query(self) -> None:
        guard = GuardrailExtension()
        guard.on_query_start("What is RAG?", {})  # should not raise

    def test_blocks_long_response(self) -> None:
        guard = GuardrailExtension(max_length=10)
        import pytest

        with pytest.raises(ValueError, match="length"):
            guard.on_query_end("A" * 11, [], 0.1, {}, 0.0)

    def test_allows_short_response(self) -> None:
        guard = GuardrailExtension(max_length=100)
        guard.on_query_end("short answer", [], 0.1, {}, 0.0)


class TestMonitoredRAGPipeline:
    def _make_mock_pipeline(self):
        pipeline = MagicMock()
        pipeline.config.top_k_final = 5
        pipeline.query.return_value = ("answer text", [])
        pipeline.ingest.return_value = 7
        pipeline.stats.return_value = {"chunks_in_store": 7}
        return pipeline

    def test_ingest_delegates_to_pipeline(self) -> None:
        pipeline = self._make_mock_pipeline()
        wrapped = MonitoredRAGPipeline(pipeline)
        count = wrapped.ingest("some/path")
        assert count == 7
        pipeline.ingest.assert_called_once_with("some/path")

    def test_query_delegates_and_returns(self) -> None:
        pipeline = self._make_mock_pipeline()
        wrapped = MonitoredRAGPipeline(pipeline)
        answer, citations = wrapped.query("What is RAG?")
        assert answer == "answer text"
        pipeline.query.assert_called_once()

    def test_stats_delegates(self) -> None:
        pipeline = self._make_mock_pipeline()
        wrapped = MonitoredRAGPipeline(pipeline)
        assert wrapped.stats() == {"chunks_in_store": 7}

    def test_otel_extension_tracks_latency_even_when_otel_down(self) -> None:
        import time

        mc = MetricsCollector(enabled=False)
        ext = OTelMetricsExtension(metrics=mc)
        # Trip the circuit breaker and keep it OPEN by setting last_failure_time to now.
        ext._cb.state = "OPEN"
        ext._cb.last_failure_time = time.monotonic()
        # Even with the breaker open, local in-process lists must still be updated.
        ext.on_query_end("ans", [], 0.55, {"prompt": 10, "completion": 5, "total": 15}, 0.001)
        assert mc.latencies == [0.55]
        assert mc.costs == [0.001]
