"""Faithfulness and quality metrics for RAG evaluation."""

from __future__ import annotations

import logging
from typing import Any, Literal

from src.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced top-level JSON object from ``text``.

    More robust than a greedy ``\\{.*\\}`` regex: walks brace depth so that
    judge responses containing multiple JSON-like fragments or stray braces
    in explanatory prose don't capture the wrong span.
    """
    import json

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


ANSWER_RELEVANCE_JUDGE_PROMPT = """You are an expert evaluator of AI answer quality. Your task is to determine whether the given answer is relevant and directly addresses the user's question.

**Question:**
{question}

**Answer:**
{answer}

**Evaluation Criteria:**
A relevant answer must:
1. Directly address the user's question
2. Stay focused on what was asked (no excessive off-topic content)
3. Be complete enough to be useful (not evasive or too vague)

**Output format — respond with a JSON object only, no other text:**
{{
    "relevant": true_or_false,
    "relevance_score": <float between 0.0 and 1.0>,
    "explanation": "Brief explanation of the score"
}}
"""


# Prompt used to evaluate faithfulness via LLM-as-judge
FAITHFULNESS_JUDGE_PROMPT = """You are an expert evaluator of AI answer faithfulness. Your task is to determine whether the given answer is fully supported by the provided context.

**Context:**
{context}

**Answer:**
{answer}

**Evaluation Criteria:**
A faithful answer must:
1. Only make claims that are directly supported by the context
2. Not contradict any information in the context
3. Not introduce external knowledge not present in the context
4. Clearly indicate when the context lacks sufficient information

**Output format — respond with a JSON object only, no other text:**
{{
    "faithful": true_or_false,
    "faithfulness_score": <float between 0.0 and 1.0>,
    "unsupported_claims": ["list of claims not supported by context"],
    "explanation": "Brief explanation of the score"
}}
"""


class FaithfulnessScorer:
    """Scores answer faithfulness by checking claims against retrieved context.

    Uses an LLM-as-judge approach: an evaluator LLM reviews the answer against
    the context and produces a faithfulness score.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: Literal["openai", "anthropic"] = "openai",
    ) -> None:
        self.model = model
        self.provider = provider
        self._client = LLMClient(provider=provider, model=model)

    def score(
        self,
        answer: str,
        contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate faithfulness of an answer against its supporting context.

        Args:
            answer: The generated answer text.
            contexts: The retrieved context chunks used to generate the answer.

        Returns:
            Dict with keys: faithful (bool), faithfulness_score (float),
            unsupported_claims (list), explanation (str).
        """
        context_text = self._format_context(contexts)
        prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context_text, answer=answer)

        response_text = self._call_judge_llm(prompt)
        result = self._parse_response(response_text)

        logger.info(
            "Faithfulness score: %.2f, faithful: %s",
            result.get("faithfulness_score", 0.0),
            result.get("faithful", False),
        )
        return result

    # ------------------------------------------------------------------
    # Provider integration
    # ------------------------------------------------------------------

    def _call_judge_llm(self, prompt: str) -> str:
        """Call the judge LLM (reuses the same provider as generation)."""
        return self._client.complete(prompt=prompt, temperature=0.0, max_tokens=512)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(contexts: list[dict[str, Any]]) -> str:
        """Flatten retrieved contexts into a single evaluable string."""
        parts = []
        for i, ctx in enumerate(contexts, start=1):
            doc = ctx.get("document", "")
            parts.append(f"[Context {i}]\n{doc}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Parse the LLM judge's JSON response, with fallback."""
        parsed = _extract_json_object(text)
        if parsed is not None:
            return parsed

        # Fallback: heuristic parsing
        logger.warning("Failed to parse judge response as JSON, using heuristic fallback")
        is_faithful = (
            "true" in text.lower() and "false" not in text.lower().split("faithful")[-1][:10]
        )
        return {
            "faithful": is_faithful,
            "faithfulness_score": 1.0 if is_faithful else 0.0,
            "unsupported_claims": [],
            "explanation": "Parsed via fallback (non-JSON response)",
        }


class AnswerRelevanceScorer:
    """Scores whether a generated answer is relevant to the original question.

    Uses an LLM-as-judge approach: an evaluator LLM reviews the question and
    answer pair and produces a relevance score between 0.0 and 1.0.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: Literal["openai", "anthropic"] = "openai",
    ) -> None:
        self.model = model
        self.provider = provider
        self._client = LLMClient(provider=provider, model=model)

    def score(self, question: str, answer: str) -> dict[str, Any]:
        """Evaluate whether the answer is relevant to the question.

        Args:
            question: The original user question.
            answer: The generated answer to evaluate.

        Returns:
            Dict with keys: relevant (bool), relevance_score (float), explanation (str).
        """
        prompt = ANSWER_RELEVANCE_JUDGE_PROMPT.format(question=question, answer=answer)
        response_text = self._call_judge_llm(prompt)
        result = self._parse_response(response_text)

        logger.info(
            "Answer relevance score: %.2f, relevant: %s",
            result.get("relevance_score", 0.0),
            result.get("relevant", False),
        )
        return result

    def _call_judge_llm(self, prompt: str) -> str:
        """Call the judge LLM."""
        return self._client.complete(prompt=prompt, temperature=0.0, max_tokens=256)

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Parse the LLM judge's JSON response, with fallback."""
        parsed = _extract_json_object(text)
        if parsed is not None:
            return parsed

        logger.warning("Failed to parse relevance judge response as JSON, using heuristic fallback")
        is_relevant = (
            "true" in text.lower() and "false" not in text.lower().split("relevant")[-1][:10]
        )
        return {
            "relevant": is_relevant,
            "relevance_score": 1.0 if is_relevant else 0.0,
            "explanation": "Parsed via fallback (non-JSON response)",
        }
