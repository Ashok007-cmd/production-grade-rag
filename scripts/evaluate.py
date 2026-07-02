#!/usr/bin/env python3
"""CLI script to run golden evaluation dataset against the RAG pipeline.

Usage:
    python scripts/evaluate.py                          # Run evaluation
    python scripts/evaluate.py --hybrid --reranker       # With Phase 2 features
    python scripts/evaluate.py --fail-on-threshold       # Exit non-zero if below threshold (CI mode)
    python scripts/evaluate.py --create-sample-dataset   # Create a sample dataset first
    python scripts/evaluate.py --provider anthropic      # Use Anthropic as eval judge
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import sys
from typing import Any

from src.config import settings
from src.evaluation.dataset import GoldenDataset
from src.evaluation.runner import EvaluationFailed, EvaluationRunner
from src.pipeline import RAGPipeline

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument(
        "--hybrid", action="store_true", help="Enable hybrid search during evaluation"
    )
    parser.add_argument("--reranker", action="store_true", help="Enable cross-encoder re-ranker")
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with non-zero code if faithfulness below threshold (CI mode)",
    )
    parser.add_argument(
        "--create-sample-dataset",
        action="store_true",
        help="Create a sample golden evaluation dataset and exit",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.faithfulness_threshold,
        help=f"Faithfulness threshold (default: {settings.faithfulness_threshold})",
    )
    parser.add_argument(
        "--export-ci-summary",
        action="store_true",
        help="Export a JSON summary for CI consumption",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=settings.llm_provider,
        help="LLM provider for the judge model (default: from RAG_LLM_PROVIDER env var)",
    )
    args = parser.parse_args()

    # Create sample dataset if requested
    if args.create_sample_dataset:
        dataset = GoldenDataset()
        sample = dataset.create_sample_dataset()
        dataset.save(sample)
        print(f"Sample dataset created with {len(sample)} examples at {dataset.path}")
        return

    # Check for required API keys before instantiating the pipeline
    provider = args.provider or settings.llm_provider
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_name):
        print(f"WARNING: Environment variable {key_name} is not set.")
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(
                "Running in CI context. Skipping evaluation because API keys are not available (e.g. pull request from fork)."
            )
            if args.export_ci_summary:
                import json

                summary = {
                    "count": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_faithfulness": 0.0,
                    "min_faithfulness": 0.0,
                    "max_faithfulness": 0.0,
                    "pass_rate": 1.0,
                    "faithful_count": 0,
                    "avg_answer_relevance": 0.0,
                    "relevant_count": 0,
                    "avg_context_precision": 0.0,
                    "avg_context_recall": 0.0,
                    "avg_contexts_used": 0.0,
                    "threshold_met": True,
                    "faithfulness_threshold": args.threshold,
                    "skipped_reason": f"Skipped because {key_name} is missing in fork PR/CI context.",
                }
                with open("eval-summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
            sys.exit(0)
        else:
            print(f"Error: {key_name} must be set to run evaluation.", file=sys.stderr)
            sys.exit(1)

    # Run evaluation
    pipeline: Any = RAGPipeline()
    if os.environ.get("MONITOR_ENABLED") == "true":
        from src.monitoring import MetricsCollector, MonitoredRAGPipeline, Tracer

        tracer = Tracer(enabled=True)
        metrics = MetricsCollector(enabled=True)
        pipeline = MonitoredRAGPipeline(pipeline, tracer=tracer, metrics=metrics)
        atexit.register(metrics.export_summary, "monitoring-summary.json")

    dataset = GoldenDataset()
    runner = EvaluationRunner(
        pipeline=pipeline,
        dataset=dataset,
        faithfulness_threshold=args.threshold,
        eval_provider=args.provider,
    )

    try:
        results = runner.run(
            use_hybrid=args.hybrid,
            use_reranker=args.reranker,
            fail_on_threshold=args.fail_on_threshold,
        )
    except EvaluationFailed as e:
        print(f"EVALUATION FAILED: {e}")
        sys.exit(1)

    if results:
        runner.print_report(results)
        if args.export_ci_summary:
            runner.export_ci_summary(results)
    else:
        print("No evaluation examples found. Use --create-sample-dataset to create some.")
        sys.exit(1)


if __name__ == "__main__":
    main()
