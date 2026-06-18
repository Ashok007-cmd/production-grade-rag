"""Tests for the evaluation framework (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.dataset import EvalExample, GoldenDataset
from src.evaluation.metrics import AnswerRelevanceScorer, FaithfulnessScorer
from src.evaluation.runner import EvaluationFailed, EvaluationRunner


class TestEvalExample:
    """Tests for the EvalExample dataclass."""

    def test_basic_creation(self) -> None:
        example = EvalExample(
            question="What is RAG?",
            reference_answer="RAG is a retrieval-augmented generation framework.",
        )
        assert example.question == "What is RAG?"
        assert example.reference_answer
        assert example.id  # auto-generated

    def test_id_generation_deterministic(self) -> None:
        ex1 = EvalExample(question="What is RAG?", reference_answer="A.")
        ex2 = EvalExample(question="What is RAG?", reference_answer="B.")
        assert ex1.id == ex2.id  # same question -> same hash

    def test_id_generation_different_questions(self) -> None:
        ex1 = EvalExample(question="What is RAG?", reference_answer="A.")
        ex2 = EvalExample(question="What is hybrid search?", reference_answer="B.")
        assert ex1.id != ex2.id

    def test_with_expected_sources(self) -> None:
        example = EvalExample(
            question="Test?",
            reference_answer="Test answer.",
            expected_sources=["doc1.txt", "doc2.txt"],
        )
        assert len(example.expected_sources) == 2

    def test_custom_id(self) -> None:
        example = EvalExample(
            question="Test?",
            reference_answer="Test.",
            id="custom_001",
        )
        assert example.id == "custom_001"


class TestGoldenDataset:
    """Tests for the GoldenDataset class."""

    def test_empty_dataset(self) -> None:
        dataset = GoldenDataset(path=Path("/tmp/nonexistent_dataset.jsonl"))
        assert len(dataset) == 0

    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "test_dataset.jsonl"
        dataset = GoldenDataset(path=path)
        examples = [
            EvalExample(question="Q1?", reference_answer="A1."),
            EvalExample(question="Q2?", reference_answer="A2."),
        ]
        dataset.save(examples)
        assert path.exists()

        # Load into a new instance
        dataset2 = GoldenDataset(path=path)
        loaded = dataset2.load()
        assert len(loaded) == 2
        assert loaded[0].question == "Q1?"
        assert loaded[1].question == "Q2?"

    def test_add_example(self, tmp_path: Path) -> None:
        path = tmp_path / "add_test.jsonl"
        dataset = GoldenDataset(path=path)
        example = EvalExample(question="Q?", reference_answer="A.")
        dataset.add(example)
        assert len(dataset) == 1

    def test_create_sample_dataset(self) -> None:
        samples = GoldenDataset.create_sample_dataset()
        assert len(samples) > 0
        for s in samples:
            assert s.question
            assert s.reference_answer

    def test_index_access(self, tmp_path: Path) -> None:
        path = tmp_path / "index_test.jsonl"
        dataset = GoldenDataset(path=path)
        examples = [
            EvalExample(question="Q1?", reference_answer="A1."),
            EvalExample(question="Q2?", reference_answer="A2."),
        ]
        dataset.save(examples)
        assert dataset[0].question == "Q1?"
        assert dataset[1].question == "Q2?"


class TestFaithfulnessScorer:
    """Tests for the FaithfulnessScorer."""

    def test_scorer_importable(self) -> None:
        scorer = FaithfulnessScorer(model="gpt-4o-mini")
        assert scorer is not None
        assert scorer.model == "gpt-4o-mini"

    def test_format_context(self) -> None:
        contexts = [
            {"document": "RAG combines retrieval and generation.", "metadata": {}},
            {"document": "Hybrid search uses BM25 and vectors.", "metadata": {}},
        ]
        formatted = FaithfulnessScorer._format_context(contexts)
        assert "[Context 1]" in formatted
        assert "[Context 2]" in formatted
        assert "RAG combines" in formatted
        assert "BM25" in formatted

    def test_parse_response_json(self) -> None:
        response = '{"faithful": true, "faithfulness_score": 0.95, "unsupported_claims": [], "explanation": "All claims supported."}'
        result = FaithfulnessScorer._parse_response(response)
        assert result["faithful"] is True
        assert result["faithfulness_score"] == 0.95

    def test_parse_response_fallback(self) -> None:
        response = "The answer is faithful and supported by the context."
        result = FaithfulnessScorer._parse_response(response)
        assert "faithful" in result

    def test_parse_response_unfaithful(self) -> None:
        response = '{"faithful": false, "faithfulness_score": 0.2, "unsupported_claims": ["Claim about X"], "explanation": "Claim X not in context."}'
        result = FaithfulnessScorer._parse_response(response)
        assert result["faithful"] is False
        assert result["faithfulness_score"] == 0.2
        assert len(result["unsupported_claims"]) == 1


class TestAnswerRelevanceScorer:
    """Tests for the AnswerRelevanceScorer."""

    def test_scorer_importable(self) -> None:
        scorer = AnswerRelevanceScorer(model="gpt-4o-mini")
        assert scorer is not None
        assert scorer.model == "gpt-4o-mini"

    def test_parse_response_json_relevant(self) -> None:
        response = '{"relevant": true, "relevance_score": 0.95, "explanation": "Directly addresses the question."}'
        result = AnswerRelevanceScorer._parse_response(response)
        assert result["relevant"] is True
        assert result["relevance_score"] == 0.95

    def test_parse_response_json_irrelevant(self) -> None:
        response = '{"relevant": false, "relevance_score": 0.1, "explanation": "Answer does not address the question."}'
        result = AnswerRelevanceScorer._parse_response(response)
        assert result["relevant"] is False
        assert result["relevance_score"] == 0.1

    def test_parse_response_fallback(self) -> None:
        response = "The answer is relevant to the question asked."
        result = AnswerRelevanceScorer._parse_response(response)
        assert "relevant" in result

    def test_score_mocked(self) -> None:
        from unittest.mock import patch

        scorer = AnswerRelevanceScorer(provider="openai")
        mock_response = '{"relevant": true, "relevance_score": 0.9, "explanation": "Answer addresses the question."}'

        with patch.object(scorer, "_call_judge_llm", return_value=mock_response) as mock_call:
            result = scorer.score(
                "What is RAG?", "RAG is a framework combining retrieval and generation."
            )
            assert result["relevant"] is True
            assert result["relevance_score"] == 0.9
            mock_call.assert_called_once()

    def test_unsupported_provider_raises(self) -> None:
        scorer = AnswerRelevanceScorer(provider="unsupported")  # type: ignore
        with pytest.raises(ValueError, match="Unsupported.*provider"):
            scorer._call_judge_llm("some prompt")


class TestEvaluationRunner:
    """Tests for the EvaluationRunner."""

    def test_runner_initialization(self) -> None:
        runner = EvaluationRunner(faithfulness_threshold=0.7)
        assert runner.faithfulness_threshold == 0.7

    def test_summarize_empty(self) -> None:
        summary = EvaluationRunner.summarize([])
        assert summary["count"] == 0

    def test_summarize_with_results(self) -> None:
        from src.evaluation.runner import EvalResult

        results = [
            EvalResult(
                example_id="1",
                question="Q1?",
                generated_answer="A1",
                reference_answer="R1",
                faithfulness_score=0.9,
                answer_relevance_score=0.8,
                passed=True,
            ),
            EvalResult(
                example_id="2",
                question="Q2?",
                generated_answer="A2",
                reference_answer="R2",
                faithfulness_score=0.5,
                answer_relevance_score=0.6,
                passed=False,
            ),
        ]
        summary = EvaluationRunner.summarize(results)
        assert summary["count"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["avg_faithfulness"] == 0.7
        assert summary["avg_answer_relevance"] == 0.7

    def test_check_threshold(self) -> None:
        from src.evaluation.runner import EvalResult

        runner = EvaluationRunner(faithfulness_threshold=0.7)
        passing_results = [
            EvalResult(
                example_id="1",
                question="Q?",
                generated_answer="A",
                reference_answer="R",
                faithfulness_score=0.8,
                answer_relevance_score=0.9,
                passed=True,
            ),
            EvalResult(
                example_id="2",
                question="Q?",
                generated_answer="A",
                reference_answer="R",
                faithfulness_score=0.9,
                answer_relevance_score=0.8,
                passed=True,
            ),
        ]
        assert runner._check_threshold(passing_results) is True

        failing_results = [
            EvalResult(
                example_id="1",
                question="Q?",
                generated_answer="A",
                reference_answer="R",
                faithfulness_score=0.3,
                answer_relevance_score=0.5,
                passed=False,
            ),
            EvalResult(
                example_id="2",
                question="Q?",
                generated_answer="A",
                reference_answer="R",
                faithfulness_score=0.4,
                answer_relevance_score=0.6,
                passed=False,
            ),
        ]
        assert runner._check_threshold(failing_results) is False

    def test_evaluation_failed_exception(self) -> None:
        with pytest.raises(EvaluationFailed):
            raise EvaluationFailed("Quality threshold not met")

    def test_faithfulness_scorer_score_mocked(self) -> None:
        """Verify FaithfulnessScorer score calling and parsing with mock judge LLM."""
        from unittest.mock import patch

        scorer = FaithfulnessScorer(provider="openai")
        mock_response = '{"faithful": true, "faithfulness_score": 1.0, "unsupported_claims": [], "explanation": "Fully grounded."}'

        with patch.object(scorer, "_call_judge_llm", return_value=mock_response) as mock_call:
            contexts = [{"document": "Retrieval-Augmented Generation is a framework."}]
            result = scorer.score("RAG is a framework.", contexts)
            assert result["faithful"] is True
            assert result["faithfulness_score"] == 1.0
            mock_call.assert_called_once()

    def test_evaluation_runner_run_mocked(self) -> None:
        """Verify EvaluationRunner end-to-end execution and full context propagation with mocked query and scorer."""
        from unittest.mock import MagicMock

        from src.evaluation.dataset import EvalExample

        runner = EvaluationRunner(faithfulness_threshold=0.7)
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = (
            "Generated answer.",
            [
                MagicMock(
                    chunk_id="c1",
                    source="doc1.txt",
                    filename="doc1.txt",
                    text_snippet="Full document text",
                    score=0.9,
                )
            ],
        )

        runner.pipeline = mock_pipeline

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {
            "faithful": True,
            "faithfulness_score": 0.8,
            "unsupported_claims": [],
            "explanation": "Ok",
        }
        runner.scorer = mock_scorer

        mock_relevance_scorer = MagicMock()
        mock_relevance_scorer.score.return_value = {
            "relevant": True,
            "relevance_score": 0.9,
            "explanation": "On-topic.",
        }
        runner.relevance_scorer = mock_relevance_scorer

        # Override examples
        runner.dataset._examples = [
            EvalExample(question="What is RAG?", reference_answer="RAG is good.")
        ]

        results = runner.run(use_hybrid=False, use_reranker=False, fail_on_threshold=True)
        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].faithfulness_score == 0.8
        assert results[0].answer_relevance_score == 0.9
        mock_pipeline.query.assert_called_once_with(
            "What is RAG?", use_hybrid=False, use_reranker=False
        )
        mock_scorer.score.assert_called_once()
        mock_relevance_scorer.score.assert_called_once()
        # Verify the context passed to the faithfulness scorer had "Full document text"
        contexts_passed = mock_scorer.score.call_args[0][1]
        assert contexts_passed[0]["document"] == "Full document text"

    def test_faithfulness_scorer_anthropic_mocked(self) -> None:
        """Verify FaithfulnessScorer scoring and Anthropic messages API integration using mock responses."""
        from unittest.mock import patch

        scorer = FaithfulnessScorer(provider="anthropic", model="claude-3-haiku-20240307")
        mock_response = '{"faithful": true, "faithfulness_score": 0.9, "unsupported_claims": [], "explanation": "Ok."}'

        with patch.object(scorer, "_call_judge_llm", return_value=mock_response) as mock_call:
            contexts = [{"document": "Retrieval-Augmented Generation."}]
            result = scorer.score("RAG is nice.", contexts)
            assert result["faithful"] is True
            assert result["faithfulness_score"] == 0.9
            mock_call.assert_called_once()

    def test_evaluation_runner_reporting_and_ci(self, tmp_path: Path) -> None:
        """Verify EvaluationRunner reports summary and saves CI files correctly."""
        from src.evaluation.runner import EvalResult

        runner = EvaluationRunner(results_dir=tmp_path)

        results = [
            EvalResult(
                example_id="1",
                question="Q1?",
                generated_answer="A1",
                reference_answer="R1",
                faithfulness_score=0.8,
                answer_relevance_score=0.9,
                passed=True,
                contexts_used=1,
            ),
            EvalResult(
                example_id="2",
                question="Q2?",
                generated_answer="A2",
                reference_answer="R2",
                faithfulness_score=0.4,
                answer_relevance_score=0.7,
                passed=False,
                contexts_used=2,
            ),
        ]

        runner.print_report(results)

        summary_path = tmp_path / "ci-summary.json"
        runner.export_ci_summary(results, output_path=str(summary_path))
        assert summary_path.exists()

        runner._save_results(results)
        files = list(tmp_path.glob("eval_*.json"))
        assert len(files) == 1

    def test_parse_response_invalid_json_fallback(self) -> None:
        """Verify heuristic fallback logic in scorer when judge returns non-JSON text."""
        response = "The judge says: this response is not faithful. The explanation is..."
        result = FaithfulnessScorer._parse_response(response)
        assert result["faithful"] is False
        assert result["faithfulness_score"] == 0.0

        response_faithful = "The judge says: true, this is faithful."
        result_faithful = FaithfulnessScorer._parse_response(response_faithful)
        assert result_faithful["faithful"] is True
        assert result_faithful["faithfulness_score"] == 1.0
