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


def _format_context(contexts: list[dict[str, Any]]) -> str:
    """Flatten retrieved contexts into a single evaluable string, numbered by rank."""
    parts = []
    for i, ctx in enumerate(contexts, start=1):
        doc = ctx.get("document", "")
        parts.append(f"[Context {i}]\n{doc}")
    return "\n\n".join(parts)


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
        return _format_context(contexts)

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


# Prompt used to judge per-chunk relevance for Context Precision
CONTEXT_PRECISION_JUDGE_PROMPT = """You are an expert evaluator of retrieval quality for a RAG system. For each retrieved context chunk below, determine whether it is relevant to answering the question — i.e. whether it contains information that would help answer the question.

**Question:**
{question}

**Retrieved Chunks (in retrieval rank order):**
{numbered_chunks}

**Output format — respond with a JSON object only, no other text:**
{{
    "verdicts": [true_or_false, ...],
    "explanation": "Brief explanation of the verdicts"
}}

The "verdicts" array must have exactly {num_chunks} entries, one per chunk, in the same order as listed above.
"""


# Prompt used to judge statement-level attribution for Context Recall
CONTEXT_RECALL_JUDGE_PROMPT = """You are an expert evaluator of retrieval completeness for a RAG system. Break the reference answer below into distinct factual statements, then determine whether each statement can be attributed to (is supported by) the retrieved context.

**Reference Answer:**
{reference_answer}

**Retrieved Context:**
{context}

**Output format — respond with a JSON object only, no other text:**
{{
    "statements": [
        {{"statement": "<extracted statement>", "attributed": true_or_false}}
    ],
    "explanation": "Brief explanation"
}}
"""


class ContextPrecisionScorer:
    """Scores retrieval precision: are the retrieved chunks relevant, and are
    relevant chunks ranked higher than irrelevant ones?

    Computes RAGAS-style Context Precision (Average Precision @ k): an LLM
    judges each retrieved chunk relevant/irrelevant, and the score rewards
    relevant chunks appearing earlier in the ranking. Unlike
    FaithfulnessScorer/AnswerRelevanceScorer, the aggregate score is computed
    deterministically from the binary per-chunk verdicts here rather than
    trusting a self-reported LLM score — more robust to judge arithmetic
    mistakes, and matches RAGAS's actual methodology.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: Literal["openai", "anthropic"] = "openai",
    ) -> None:
        self.model = model
        self.provider = provider
        self._client = LLMClient(provider=provider, model=model)

    def score(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Score retrieval precision for a question against its retrieved contexts.

        Returns a dict with keys: context_precision_score (float),
        verdicts (list[bool], one per chunk), explanation (str).
        """
        if not contexts:
            return {
                "context_precision_score": 0.0,
                "verdicts": [],
                "explanation": "No contexts were retrieved.",
            }

        numbered_chunks = "\n\n".join(
            f"[{i}] {ctx.get('document', '')}" for i, ctx in enumerate(contexts, start=1)
        )
        prompt = CONTEXT_PRECISION_JUDGE_PROMPT.format(
            question=question, numbered_chunks=numbered_chunks, num_chunks=len(contexts)
        )
        response_text = self._call_judge_llm(prompt)
        parsed = _extract_json_object(response_text)
        verdicts = self._normalize_verdicts(parsed, len(contexts))
        score = self._average_precision(verdicts)

        logger.info("Context precision score: %.2f (%d chunks)", score, len(contexts))
        return {
            "context_precision_score": score,
            "verdicts": verdicts,
            "explanation": parsed.get("explanation", "")
            if parsed
            else "Failed to parse judge response as JSON.",
        }

    def _call_judge_llm(self, prompt: str) -> str:
        return self._client.complete(prompt=prompt, temperature=0.0, max_tokens=512)

    @staticmethod
    def _normalize_verdicts(parsed: dict[str, Any] | None, expected_len: int) -> list[bool]:
        """Coerce the judge's verdicts into a fixed-length bool list.

        Missing or unparseable verdicts default to False (conservative:
        an unjudgeable chunk is not counted as relevant).
        """
        verdicts = parsed.get("verdicts") if parsed else None
        if not isinstance(verdicts, list):
            return [False] * expected_len
        normalized = [bool(v) for v in verdicts[:expected_len]]
        normalized += [False] * (expected_len - len(normalized))
        return normalized

    @staticmethod
    def _average_precision(verdicts: list[bool]) -> float:
        """Compute Average Precision @ k over a ranked list of relevance verdicts."""
        relevant_count = 0
        precision_sum = 0.0
        for i, is_relevant in enumerate(verdicts, start=1):
            if is_relevant:
                relevant_count += 1
                precision_sum += relevant_count / i
        if relevant_count == 0:
            return 0.0
        return round(precision_sum / relevant_count, 4)


class ContextRecallScorer:
    """Scores retrieval completeness: does the retrieved context cover
    everything needed to reconstruct the reference (golden) answer?

    Computes RAGAS-style Context Recall: the reference answer is broken
    into factual statements, each classified as attributable to the
    retrieved context or not; recall = (# attributable) / (total statements).
    Requires a golden-dataset reference answer, unlike faithfulness/
    relevance which only need the generated answer.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: Literal["openai", "anthropic"] = "openai",
    ) -> None:
        self.model = model
        self.provider = provider
        self._client = LLMClient(provider=provider, model=model)

    def score(self, reference_answer: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Score retrieval recall for a reference answer against retrieved contexts.

        Returns a dict with keys: context_recall_score (float),
        statements (list[dict] with "statement"/"attributed" keys), explanation (str).
        """
        context_text = _format_context(contexts)
        prompt = CONTEXT_RECALL_JUDGE_PROMPT.format(
            reference_answer=reference_answer, context=context_text
        )
        response_text = self._call_judge_llm(prompt)
        parsed = _extract_json_object(response_text)
        statements = self._normalize_statements(parsed)

        if not statements:
            return {
                "context_recall_score": 0.0,
                "statements": [],
                "explanation": "Could not extract statements from the reference answer.",
            }

        attributed_count = sum(1 for s in statements if s["attributed"])
        score = round(attributed_count / len(statements), 4)

        logger.info("Context recall score: %.2f (%d statements)", score, len(statements))
        return {
            "context_recall_score": score,
            "statements": statements,
            "explanation": parsed.get("explanation", "")
            if parsed
            else "Failed to parse judge response as JSON.",
        }

    def _call_judge_llm(self, prompt: str) -> str:
        return self._client.complete(prompt=prompt, temperature=0.0, max_tokens=768)

    @staticmethod
    def _normalize_statements(parsed: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Coerce the judge's statement classifications into a clean list."""
        raw_statements = parsed.get("statements") if parsed else None
        if not isinstance(raw_statements, list):
            return []
        normalized = []
        for item in raw_statements:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "statement": str(item.get("statement", "")),
                    "attributed": bool(item.get("attributed", False)),
                }
            )
        return normalized
