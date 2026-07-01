"""Tests for the FastAPI HTTP service layer (src/api/app.py)."""

from __future__ import annotations

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
