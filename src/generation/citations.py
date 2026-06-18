"""Source citation formatting and tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Citation:
    """A single source citation with locator information."""

    chunk_id: str
    source: str
    filename: str
    text_snippet: str
    score: float = 0.0


class CitationFormatter:
    """Formats source citations alongside LLM-generated answers."""

    @staticmethod
    def build_citations(results: list[dict[str, Any]]) -> list[Citation]:
        """Build Citation objects from retrieval results."""
        citations: list[Citation] = []
        for i, r in enumerate(results):
            meta = r.get("metadata", {})
            score = r.get("rerank_score")
            if score is None:
                score = r.get("score")
            if score is None:
                score = 0.0
            citations.append(
                Citation(
                    chunk_id=r.get("id", f"chunk-{i}"),
                    source=meta.get("source", str(meta)),
                    filename=meta.get("filename", "unknown"),
                    text_snippet=r.get("document", ""),
                    score=score,
                )
            )
        return citations

    @staticmethod
    def format_answer_with_citations(
        answer: str,
        citations: list[Citation],
        format: str = "inline",
    ) -> str:
        """Format an answer with source citations appended.

        Args:
            answer: The LLM-generated answer text.
            citations: List of Citation objects.
            format: "inline" (bracketed numbers) or "endnote" (numbered list at bottom).

        Returns:
            Formatted answer string with citations.
        """
        result = answer.strip()

        if format == "endnote":
            if citations:
                result += "\n\n---\n**Sources:**\n"
                for i, c in enumerate(citations, start=1):
                    snippet = c.text_snippet.replace("\n", " ")[:150]
                    result += f"{i}. {c.filename} — *{snippet}...*\n"
        elif format == "inline":
            if citations:
                result += "\n\n**Citations:**\n"
                for i, c in enumerate(citations, start=1):
                    result += f"[{i}] {c.filename}\n"

        return result

    @staticmethod
    def to_dict(citations: list[Citation]) -> list[dict[str, Any]]:
        """Serialize citations to a JSON-compatible dict list."""
        return [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "filename": c.filename,
                "text_snippet": c.text_snippet,
                "score": c.score,
            }
            for c in citations
        ]
