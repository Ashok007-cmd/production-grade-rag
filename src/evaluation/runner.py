"""Evaluation runner — executes golden dataset against the pipeline and reports results."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from src.evaluation.dataset import EvalExample, GoldenDataset
from src.evaluation.metrics import AnswerRelevanceScorer, FaithfulnessScorer
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single example against the pipeline."""

    example_id: str
    question: str
    generated_answer: str
    reference_answer: str
    faithfulness_score: float = 0.0
    is_faithful: bool = False
    unsupported_claims: list[str] = field(default_factory=list)
    answer_relevance_score: float = 0.0
    is_relevant: bool = False
    contexts_used: int = 0
    contexts: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False


class EvaluationRunner:
    """Runs the golden evaluation dataset against the RAG pipeline and enforces quality gates.

    Designed for CI integration: returns a non-zero exit code when quality
    drops below the configured threshold.
    """

    def __init__(
        self,
        pipeline: RAGPipeline | None = None,
        dataset: GoldenDataset | None = None,
        faithfulness_threshold: float = 0.7,
        eval_model: str = "gpt-4o-mini",
        eval_provider: Literal["openai", "anthropic"] = "openai",
        results_dir: Path | str | None = None,
        max_workers: int = 4,
    ) -> None:
        self.pipeline = pipeline or RAGPipeline()
        self.dataset = dataset or GoldenDataset()
        self.faithfulness_threshold = faithfulness_threshold
        self.eval_model = eval_model
        self.eval_provider = eval_provider
        self.results_dir = Path(results_dir) if results_dir else Path("data/eval_results")
        self.max_workers = max_workers

        self.scorer = FaithfulnessScorer(model=eval_model, provider=eval_provider)
        self.relevance_scorer = AnswerRelevanceScorer(model=eval_model, provider=eval_provider)

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------

    def run(
        self,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        fail_on_threshold: bool = False,
    ) -> list[EvalResult]:
        """Run the full evaluation suite.

        Args:
            use_hybrid: Enable hybrid search (Phase 2).
            use_reranker: Enable cross-encoder re-ranking (Phase 2).
            fail_on_threshold: If True, raise EvaluationFailed when quality
                              drops below threshold (for CI use).

        Returns:
            List of EvalResult objects.

        Raises:
            EvaluationFailed: When fail_on_threshold=True and scores are below threshold.
        """
        examples = self.dataset.examples
        if not examples:
            logger.warning("No evaluation examples found. Load the dataset first.")
            return []

        if self.max_workers > 1 and len(examples) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(
                    pool.map(
                        lambda ex: self._evaluate_single(ex, use_hybrid, use_reranker),
                        examples,
                    )
                )
        else:
            results = [
                self._evaluate_single(example, use_hybrid, use_reranker) for example in examples
            ]

        # Save results
        self._save_results(results)

        # Report summary
        summary = self.summarize(results)
        logger.info("Evaluation complete: %s", json.dumps(summary, indent=2))

        # Enforce quality gate
        if fail_on_threshold and not self._check_threshold(results):
            raise EvaluationFailed(
                f"Faithfulness threshold ({self.faithfulness_threshold}) not met. "
                f"Average: {summary['avg_faithfulness']:.2f}, "
                f"Pass rate: {summary['pass_rate']:.1%}"
            )

        return results

    def _evaluate_single(
        self,
        example: EvalExample,
        use_hybrid: bool,
        use_reranker: bool,
    ) -> EvalResult:
        """Evaluate a single example."""
        # Generate answer
        answer, citations = self.pipeline.query(
            example.question,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

        # Build contexts from citations for faithfulness scoring
        contexts = [
            {"document": c.text_snippet, "metadata": {"source": c.source}} for c in citations
        ]

        # Score faithfulness
        score_result = self.scorer.score(answer, contexts)
        faithfulness_score = score_result.get("faithfulness_score", 0.0)
        is_faithful = score_result.get("faithful", False)
        unsupported = score_result.get("unsupported_claims", [])

        # Score answer relevance
        relevance_result = self.relevance_scorer.score(example.question, answer)
        answer_relevance_score = relevance_result.get("relevance_score", 0.0)
        is_relevant = relevance_result.get("relevant", False)

        return EvalResult(
            example_id=example.id,
            question=example.question,
            generated_answer=answer,
            reference_answer=example.reference_answer,
            faithfulness_score=faithfulness_score,
            is_faithful=is_faithful,
            unsupported_claims=unsupported,
            answer_relevance_score=answer_relevance_score,
            is_relevant=is_relevant,
            contexts_used=len(citations),
            contexts=contexts,
            passed=faithfulness_score >= self.faithfulness_threshold,
        )

    # ------------------------------------------------------------------
    # Results & Reporting
    # ------------------------------------------------------------------

    def _save_results(self, results: list[EvalResult]) -> None:
        """Persist evaluation results to a timestamped JSON file."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.results_dir / f"eval_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "threshold": self.faithfulness_threshold,
            "results": [asdict(r) for r in results],
            "summary": self.summarize(results),
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", path)

    @staticmethod
    def summarize(results: list[EvalResult]) -> dict[str, Any]:
        """Compute aggregate statistics over evaluation results."""
        if not results:
            return {"count": 0}

        faith_scores = [r.faithfulness_score for r in results]
        rel_scores = [r.answer_relevance_score for r in results]
        passed = sum(1 for r in results if r.passed)
        faithful = sum(1 for r in results if r.is_faithful)
        relevant = sum(1 for r in results if r.is_relevant)

        return {
            "count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 4),
            "min_faithfulness": round(min(faith_scores), 4),
            "max_faithfulness": round(max(faith_scores), 4),
            "pass_rate": round(passed / len(results), 4),
            "faithful_count": faithful,
            "avg_answer_relevance": round(sum(rel_scores) / len(rel_scores), 4),
            "relevant_count": relevant,
            "avg_contexts_used": round(sum(r.contexts_used for r in results) / len(results), 1),
        }

    def _check_threshold(self, results: list[EvalResult]) -> bool:
        """Check if aggregate scores meet the quality threshold."""
        summary = self.summarize(results)
        avg = summary["avg_faithfulness"]
        return avg >= self.faithfulness_threshold

    def print_report(self, results: list[EvalResult]) -> None:
        """Print a human-readable evaluation report to stdout."""
        summary = self.summarize(results)
        print("=" * 60)
        print("RAG EVALUATION REPORT")
        print("=" * 60)
        print(f"  Examples evaluated:  {summary['count']}")
        print(f"  Passed:              {summary['passed']}")
        print(f"  Failed:              {summary['failed']}")
        print(f"  Pass rate:           {summary['pass_rate']:.1%}")
        print(f"  Avg faithfulness:    {summary['avg_faithfulness']:.4f}")
        print(f"  Min faithfulness:    {summary['min_faithfulness']:.4f}")
        print(f"  Max faithfulness:    {summary['max_faithfulness']:.4f}")
        print(f"  Avg answer relevance:{summary['avg_answer_relevance']:.4f}")
        print(f"  Threshold:           {self.faithfulness_threshold}")
        print()

        for i, r in enumerate(results, 1):
            status = "PASS" if r.passed else "FAIL"
            print(f"  {i}. [{status}] Q: {r.question[:60]}...")
            print(
                f"     Faithfulness: {r.faithfulness_score:.4f}  Relevance: {r.answer_relevance_score:.4f}"
            )
            print()

    def export_ci_summary(
        self, results: list[EvalResult], output_path: str = "eval-summary.json"
    ) -> None:
        """Export a minimal CI-friendly summary (used by GitHub Actions)."""
        summary = self.summarize(results)
        summary["threshold_met"] = self._check_threshold(results)
        summary["faithfulness_threshold"] = self.faithfulness_threshold
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("CI summary exported to %s", output_path)


class EvaluationFailed(Exception):
    """Raised when evaluation scores fall below the configured threshold."""

    pass
