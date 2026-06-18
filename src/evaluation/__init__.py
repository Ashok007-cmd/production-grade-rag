"""Evaluation — golden dataset, faithfulness metrics, CI runner."""

from src.evaluation.dataset import EvalExample, GoldenDataset
from src.evaluation.metrics import FaithfulnessScorer
from src.evaluation.runner import EvalResult, EvaluationRunner

__all__ = ["GoldenDataset", "EvalExample", "FaithfulnessScorer", "EvaluationRunner", "EvalResult"]
