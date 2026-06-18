"""LLM response generation with context injection and source citation."""

from __future__ import annotations

import logging
from typing import Any, Literal

from src.generation.llm_client import LLMClient
from src.utils.i18n import _

logger = logging.getLogger(__name__)

# Default system prompt for the RAG generator
DEFAULT_SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question based ONLY on the provided context. If the context does not contain sufficient information to answer, state that clearly — do not make up information.

For every claim you make, cite the source using the numbered references in brackets like [1], [2], etc. Each source maps to the corresponding context chunk provided below.

**Context:**
{context}

**Instructions:**
1. Answer concisely and accurately.
2. Cite sources for every factual claim using [number] notation.
3. If the context doesn't contain the answer, say "I cannot find sufficient information in the provided documents to answer this question."
4. Do NOT use external knowledge — only the provided context.
"""


class Generator:
    """Generates answers using an LLM, with retrieved context injection.

    Supports OpenAI and Anthropic as backends. Configure via environment variables
    (OPENAI_API_KEY or ANTHROPIC_API_KEY).
    """

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = LLMClient(provider=provider, model=model)

    def generate(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> str:
        """Generate an answer from query + retrieved contexts.

        Args:
            query: The user's question.
            contexts: Retrieved chunks (each must have "document" key).
            system_prompt: Optional override for the default system prompt.

        Returns:
            Generated answer string (with source citations).
        """
        formatted_context = self._format_context(contexts)
        prompt = (system_prompt or _(DEFAULT_SYSTEM_PROMPT)).format(context=formatted_context)

        return self._client.complete(
            prompt=query,
            system=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def generate_async(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> str:
        """Generate an answer asynchronously from query + retrieved contexts.

        Args:
            query: The user's question.
            contexts: Retrieved chunks (each must have "document" key).
            system_prompt: Optional override for the default system prompt.

        Returns:
            Generated answer string (with source citations).
        """
        formatted_context = self._format_context(contexts)
        prompt = (system_prompt or _(DEFAULT_SYSTEM_PROMPT)).format(context=formatted_context)

        return await self._client.complete_async(
            prompt=query,
            system=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(contexts: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            doc = ctx.get("document", "")
            meta = ctx.get("metadata", {})
            source = meta.get("source", meta.get("filename", f"Source {i}"))
            parts.append(f"[{i}] (Source: {source})\n{doc}\n")
        return "\n".join(parts)
