"""Cross-encoder re-ranker for improving retrieval precision."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ranks retrieval results using a cross-encoder model.

    Cross-encoders jointly encode query + document for more accurate relevance
    scoring than bi-encoder (vector similarity) approaches, at the cost of
    additional inference time per pair. Used as a final precision refinement step.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _lazy_load(self) -> None:
        """Load the cross-encoder model on first use."""
        if self._model is None:
            try:
                import torch

                torch.set_num_threads(1)
                logger.info("Constraining PyTorch to 1 CPU thread.")
            except ImportError:
                pass

            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder model: %s", self.model_name)
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for re-ranking. "
                    "Install with: pip install sentence-transformers"
                ) from None

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank retrieved results by cross-encoder relevance scoring.

        Args:
            query: The original search query.
            results: List of result dicts (must contain "document" key).
            top_k: Number of top results to return after re-ranking.

        Returns:
            Results sorted by cross-encoder relevance score (descending).
        """
        if not results:
            return []

        self._lazy_load()
        if self._model is None:
            raise RuntimeError(f"Cross-encoder model '{self.model_name}' failed to load")

        # Prepare query-document pairs
        pairs = [(query, r["document"]) for r in results]

        # Score all pairs
        scores = self._model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        # Augment results with cross-encoder scores
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = round(float(score), 4)

        # Sort by cross-encoder score descending
        reranked = sorted(results, key=lambda r: r.get("rerank_score", 0.0), reverse=True)

        logger.debug(
            "Re-ranked %d results. Top score: %.4f, bottom score: %.4f",
            len(reranked),
            reranked[0]["rerank_score"],
            reranked[-1]["rerank_score"],
        )

        return reranked[:top_k]
