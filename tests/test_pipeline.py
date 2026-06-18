"""Integration tests for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import RAGPipeline


class TestRAGPipeline:
    """Integration tests for the full RAG pipeline."""

    def test_pipeline_initialization(self) -> None:
        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.chunker is not None
        assert pipeline.vector_store is not None
        assert pipeline.generator is not None

    def test_context_budget_truncates_long_contexts(self) -> None:
        pipeline = RAGPipeline()
        pipeline.config = pipeline.config.model_copy(update={"max_context_chars": 10})

        contexts = [
            {"document": "a" * 7, "metadata": {}},
            {"document": "b" * 7, "metadata": {}},
        ]
        result = pipeline._apply_context_budget(contexts)

        assert len(result) == 2
        assert result[0]["document"] == "a" * 7
        assert result[1]["document"] == "b" * 3

    def test_context_budget_keeps_whole_chunks_within_limit(self) -> None:
        pipeline = RAGPipeline()
        pipeline.config = pipeline.config.model_copy(update={"max_context_chars": 100})

        contexts = [
            {"document": "short", "metadata": {}},
            {"document": "also short", "metadata": {}},
        ]
        result = pipeline._apply_context_budget(contexts)

        assert result == contexts

    def test_pipeline_stats(self) -> None:
        pipeline = RAGPipeline()
        stats = pipeline.stats()
        assert "chunks_in_store" in stats
        assert "embedding_model" in stats
        assert "llm_provider" in stats

    def test_pipeline_reset(self) -> None:
        pipeline = RAGPipeline()
        pipeline.reset()
        assert pipeline.vector_store.count() == 0

    def test_ingest_sample_docs(self, sample_docs_dir: Path) -> None:
        pipeline = RAGPipeline()
        pipeline.reset()
        count = pipeline.ingest(sample_docs_dir)
        assert count > 0
        assert pipeline.vector_store.count() > 0

    def test_query_with_no_context_returns_fallback(self) -> None:
        pipeline = RAGPipeline()
        pipeline.reset()
        answer, citations = pipeline.query("What is the meaning of life?")
        assert "could not find" in answer.lower() or "any relevant information" in answer.lower()

    def test_import_hybrid_retrieval(self) -> None:
        """Phase 2: Hybrid retriever importable."""
        from src.retrieval.hybrid import HybridRetriever

        assert HybridRetriever is not None

    def test_import_reranker(self) -> None:
        """Phase 2: Cross-encoder re-ranker importable."""
        from src.retrieval.reranker import CrossEncoderReranker

        assert CrossEncoderReranker is not None

    def test_import_evaluation_modules(self) -> None:
        """Phase 3: Evaluation modules importable."""
        from src.evaluation.dataset import EvalExample, GoldenDataset
        from src.evaluation.metrics import FaithfulnessScorer
        from src.evaluation.runner import EvalResult, EvaluationRunner

        assert GoldenDataset is not None
        assert EvalExample is not None
        assert FaithfulnessScorer is not None
        assert EvaluationRunner is not None
        assert EvalResult is not None

    def test_config_has_phase2_settings(self) -> None:
        """Phase 2 config values exist."""
        from src.config import settings

        assert hasattr(settings, "hybrid_alpha")
        assert hasattr(settings, "rrf_k")
        assert hasattr(settings, "reranker_model")

    def test_config_has_phase3_settings(self) -> None:
        """Phase 3 config values exist."""
        from src.config import settings

        assert hasattr(settings, "faithfulness_threshold")
        assert hasattr(settings, "eval_llm_model")

    def test_query_with_context_mocked(self, sample_docs_dir: Path) -> None:
        """Verify candidate slicing and generator call when not reranking."""
        from unittest.mock import patch

        pipeline = RAGPipeline()
        pipeline.reset()
        pipeline.ingest(sample_docs_dir)

        with patch.object(
            pipeline.generator, "generate", return_value="This is a mock answer."
        ) as mock_generate:
            answer, citations = pipeline.query(
                "RAG keyword search", use_hybrid=True, use_reranker=False, top_k=2
            )
            assert answer == "This is a mock answer."
            assert len(citations) <= 2
            mock_generate.assert_called_once()
            contexts_passed = mock_generate.call_args[0][1]
            assert len(contexts_passed) <= 2

    def test_query_with_reranker_mocked(self, sample_docs_dir: Path) -> None:
        """Verify reranker execution and candidate fetching behavior."""
        from unittest.mock import MagicMock, patch

        pipeline = RAGPipeline()
        pipeline.reset()
        pipeline.ingest(sample_docs_dir)

        # Mock the reranker to avoid loading the actual cross-encoder model
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "id": "doc1",
                "document": "RAG combines retrieval.",
                "metadata": {"source": "doc1.txt"},
                "rerank_score": 0.95,
            }
        ]

        with patch.object(pipeline, "_get_reranker", return_value=mock_reranker):
            with patch.object(
                pipeline.generator, "generate", return_value="Reranked answer."
            ) as mock_generate:
                answer, citations = pipeline.query(
                    "RAG keyword", use_hybrid=False, use_reranker=True, top_k=1
                )
                assert answer == "Reranked answer."
                assert len(citations) == 1
                mock_generate.assert_called_once()
                mock_reranker.rerank.assert_called_once()
                # Verify fetch_k was set to top_k_retrieval (default 20) during retrieval
                called_contexts = mock_reranker.rerank.call_args[0][1]
                assert len(called_contexts) > 1

    def test_citation_formatter_formats(self) -> None:
        """Verify CitationFormatter outputs correct inline, endnote, and serialized formats."""
        from src.generation.citations import Citation, CitationFormatter

        citations = [
            Citation(
                chunk_id="c1",
                source="doc1.txt",
                filename="doc1.txt",
                text_snippet="Sample document context.",
                score=0.9,
            )
        ]

        # Test inline format
        formatted_inline = CitationFormatter.format_answer_with_citations(
            "The answer text.", citations, format="inline"
        )
        assert "Citations:" in formatted_inline
        assert "[1] doc1.txt" in formatted_inline

        # Test endnote format
        formatted_endnote = CitationFormatter.format_answer_with_citations(
            "The answer text.", citations, format="endnote"
        )
        assert "Sources:" in formatted_endnote
        assert "doc1.txt — *Sample document context." in formatted_endnote

        # Test to_dict
        serialized = CitationFormatter.to_dict(citations)
        assert len(serialized) == 1
        assert serialized[0]["chunk_id"] == "c1"

    def test_generator_invalid_provider(self) -> None:
        """Verify Generator raises ValueError when initialized with unsupported provider."""
        from src.generation.generator import Generator

        generator = Generator(provider="invalid")  # type: ignore
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            generator.generate("query", [{"document": "ctx"}])

    def test_generator_anthropic_call_mocked(self) -> None:
        """Verify Anthropic provider message calling and parsing flow."""
        from unittest.mock import MagicMock, patch

        from src.generation.generator import Generator

        generator = Generator(provider="anthropic")

        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Anthropic mock response."
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message

        with patch("anthropic.Anthropic", return_value=mock_client):
            answer = generator.generate("What is RAG?", [{"document": "Retrieval framework."}])
            assert answer == "Anthropic mock response."

    def test_generator_retry_logic_mocked(self) -> None:
        """Verify generator retry wrapper catches transient exceptions and executes retries with backoff."""
        from unittest.mock import MagicMock, patch

        from src.generation.generator import Generator

        generator = Generator(provider="openai")

        mock_client = MagicMock()

        from openai import APIConnectionError

        mock_request = MagicMock()
        mock_request.url = "http://mock"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success after retry."))]

        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(message="Connection failed", request=mock_request),
            mock_response,
        ]

        with patch("openai.OpenAI", return_value=mock_client):
            with patch("time.sleep") as mock_sleep:
                answer = generator.generate("query", [{"document": "ctx"}])
                assert answer == "Success after retry."
                assert mock_client.chat.completions.create.call_count == 2
                mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_query_async_mocked(self, sample_docs_dir: Path) -> None:
        """Verify async query execution path with mocked LLM completions."""
        from unittest.mock import AsyncMock, patch

        pipeline = RAGPipeline()
        pipeline.reset()
        pipeline.ingest(sample_docs_dir)

        with patch.object(
            pipeline.generator, "generate_async", new_callable=AsyncMock
        ) as mock_generate_async:
            mock_generate_async.return_value = "This is an async mock answer."
            answer, citations = await pipeline.query_async(
                "RAG search", use_hybrid=False, use_reranker=False, top_k=2
            )
            assert answer == "This is an async mock answer."
            assert len(citations) <= 2
            mock_generate_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_client_complete_async_mocked(self) -> None:
        """Verify LLMClient.complete_async triggers AsyncOpenAI chat completion."""
        from unittest.mock import AsyncMock, patch

        from src.generation.llm_client import LLMClient

        client = LLMClient(provider="openai")
        mock_openai = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Async Success."))]
        mock_openai.chat.completions.create.return_value = mock_response

        with patch("openai.AsyncOpenAI", return_value=mock_openai):
            res = await client.complete_async("hello")
            assert res == "Async Success."
            mock_openai.chat.completions.create.assert_called_once()
