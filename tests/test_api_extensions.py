"""Tests for token usage / latency headers and provider error mapping in the RAG API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def test_query_response_usage_headers(api_client: TestClient) -> None:
    """Verify that response headers contain token usage metrics and LLM latency."""
    # Mock LLMClient.complete_async to return custom response and register usage stats
    mock_complete = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Mocked LLM response."))]
    
    # Mock token usage numbers
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 42
    mock_usage.completion_tokens = 24
    mock_response.usage = mock_usage
    
    mock_complete.return_value = mock_response

    # Patch complete_async and vector store retrieval to ensure it passes quickly
    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        
        # Async method returns mock_response
        async def mock_chat_completion(*args, **kwargs):
            return mock_response
        mock_client_instance.chat.completions.create = mock_chat_completion

        # Mock vector store retrieve
        with patch("src.pipeline.RAGPipeline._retrieve", return_value=[{"document": "test doc context", "metadata": {"source": "test.txt"}}]):
            response = api_client.post("/query", json={"question": "What is the secret of the universe?"})
            
            assert response.status_code == 200
            headers = response.headers
            
            assert "X-RAG-Prompt-Tokens" in headers
            assert "X-RAG-Completion-Tokens" in headers
            assert "X-RAG-Total-Tokens" in headers
            assert "X-RAG-LLM-Latency-Sec" in headers
            
            assert headers["X-RAG-Prompt-Tokens"] == "42"
            assert headers["X-RAG-Completion-Tokens"] == "24"
            assert headers["X-RAG-Total-Tokens"] == "66"
            assert float(headers["X-RAG-LLM-Latency-Sec"]) >= 0.0


def test_openai_rate_limit_exception_mapping(api_client: TestClient) -> None:
    """Verify that an upstream OpenAI RateLimitError maps to HTTP 429."""
    from openai import RateLimitError
    
    mock_request = MagicMock()
    mock_request.url = "https://api.openai.com/v1/chat/completions"
    
    async def mock_raise(*args, **kwargs):
        raise RateLimitError(
            message="Rate limit exceeded on OpenAI",
            response=MagicMock(),
            body=None,
        )

    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = mock_raise
        
        with patch("src.pipeline.RAGPipeline._retrieve", return_value=[{"document": "test doc", "metadata": {"source": "t.txt"}}]):
            response = api_client.post("/query", json={"question": "Trigger rate limit"})
            
            assert response.status_code == 429
            assert "rate limit exceeded" in response.json()["detail"].lower()


def test_openai_connection_exception_mapping(api_client: TestClient) -> None:
    """Verify that an upstream OpenAI APIConnectionError maps to HTTP 503."""
    from openai import APIConnectionError
    
    async def mock_raise(*args, **kwargs):
        raise APIConnectionError(message="Connection failed", request=MagicMock())

    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = mock_raise
        
        with patch("src.pipeline.RAGPipeline._retrieve", return_value=[{"document": "test doc", "metadata": {"source": "t.txt"}}]):
            response = api_client.post("/query", json={"question": "Trigger connection error"})
            
            assert response.status_code == 503
            assert "failed to connect" in response.json()["detail"].lower()
