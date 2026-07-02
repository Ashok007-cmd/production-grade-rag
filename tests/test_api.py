"""Tests for the FastAPI HTTP service layer (src/api/app.py)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_app


@pytest.fixture(autouse=True)
def _reset_pipeline_singleton() -> None:
    """Ensure each test gets a fresh pipeline built against isolated config.

    The autouse fixtures in conftest.py (isolate_chroma_path,
    mock_embedding_function) patch settings/embedding behavior per-test, so
    any cached singleton from a previous test must be dropped.
    """
    api_app.reset_pipeline()
    yield
    api_app.reset_pipeline()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_app.app)


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_empty_question_returns_400(client: TestClient) -> None:
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400


def test_query_too_long_question_returns_400(client: TestClient) -> None:
    long_question = "a" * 2001
    response = client.post("/query", json={"question": long_question})
    assert response.status_code == 400


def test_ingest_nonexistent_path_returns_400(client: TestClient) -> None:
    response = client.post("/ingest", json={"source": "/no/such/path/at/all", "reset": False})
    assert response.status_code == 400


def test_ingest_and_stats(client: TestClient, sample_docs_dir: Path) -> None:
    response = client.post("/ingest", json={"source": str(sample_docs_dir), "reset": True})
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_ingested"] > 0
    assert body["total_chunks"] >= body["chunks_ingested"]

    stats_response = client.get("/stats")
    assert stats_response.status_code == 200
    stats_body = stats_response.json()
    assert stats_body["chunks_in_store"] == body["total_chunks"]


def test_query_after_ingest_returns_answer_and_citations(
    client: TestClient, sample_docs_dir: Path
) -> None:
    ingest_response = client.post("/ingest", json={"source": str(sample_docs_dir), "reset": True})
    assert ingest_response.status_code == 200

    pipeline = api_app.get_pipeline()
    from unittest.mock import AsyncMock

    with patch.object(pipeline.generator, "generate_async", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "This is a mock answer."
        response = client.post("/query", json={"question": "What is RAG?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "citations" in body
    assert body["answer"] == "This is a mock answer."
    assert isinstance(body["citations"], list)


def _poll_ingest_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/ingest/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"Ingest job {job_id} did not reach a terminal state in {timeout}s")


def test_ingest_async_returns_job_id_and_completes(sample_docs_dir: Path) -> None:
    # A context-managed TestClient keeps one event loop/portal alive across
    # calls, so the background asyncio.create_task ingestion job actually
    # gets scheduled between the POST and the polling GETs below. Without
    # `with`, each call can run on its own short-lived loop and the task
    # never progresses — a TestClient quirk, not a real-server behavior
    # (verified separately against a live uvicorn process).
    with TestClient(api_app.app) as client:
        response = client.post(
            "/ingest/async", json={"source": str(sample_docs_dir), "reset": True}
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "pending"
        job_id = body["job_id"]

        final = _poll_ingest_job(client, job_id)
        assert final["status"] == "completed"
        assert final["chunks_ingested"] > 0
        assert final["total_chunks"] >= final["chunks_ingested"]
        assert final["error"] is None


def test_ingest_async_invalid_path_fails_fast_not_as_job(client: TestClient) -> None:
    response = client.post("/ingest/async", json={"source": "/no/such/path", "reset": False})
    assert response.status_code == 400


def test_ingest_job_status_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get("/ingest/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_ingest_async_job_records_failure(
    sample_docs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = api_app.get_pipeline()

    def boom(source) -> int:
        raise ValueError("simulated ingestion failure")

    monkeypatch.setattr(pipeline, "ingest", boom)

    with TestClient(api_app.app) as client:
        response = client.post(
            "/ingest/async", json={"source": str(sample_docs_dir), "reset": False}
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        final = _poll_ingest_job(client, job_id)
        assert final["status"] == "failed"
        assert "simulated ingestion failure" in final["error"]


def test_metrics_returns_prometheus_text_and_reflects_requests(client: TestClient) -> None:
    client.get("/healthz")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "rag_http_requests_total" in body
    assert "rag_http_request_duration_seconds" in body
    assert 'path="/healthz"' in body


def test_readyz_returns_ok(client: TestClient) -> None:
    pipeline = api_app.get_pipeline()
    with patch.object(pipeline, "_get_hybrid_retriever") as mock_hybrid:
        with patch.object(pipeline, "_get_reranker") as mock_reranker:
            response = client.get("/readyz")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            assert mock_hybrid.call_count >= 1
            mock_reranker.assert_called_once()
