"""Hybrid retrieval combining BM25 keyword search with vector similarity."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuses BM25 (keyword) and vector (semantic) search results using RRF.

    Reciprocal Rank Fusion (RRF) combines rankings from multiple methods
    into a single relevance score, smoothing out individual method weaknesses.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        alpha: float = 0.6,
        rrf_k: int = 60,
        persist_path: str | Path | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")

        self.vector_store = vector_store
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.persist_path = (
            Path(persist_path)
            if persist_path
            else vector_store.persist_path.parent
            / f"bm25_index_{vector_store.collection_name}.json"
        )
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metadatas: list[dict[str, Any]] = []
        self._stale = False  # Start False to allow loading from disk on query.
        # If disk load fails or misses, search will trigger build_index.

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Build the BM25 index from all chunks in the vector store."""
        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            logger.warning("No chunks found to build BM25 index")
            self._bm25 = BM25Okapi(corpus=[[""]])
            self._corpus_ids = []
            self._corpus_texts = []
            self._corpus_metadatas = []
            self._save_index()
            return

        self._corpus_ids = [c["id"] for c in all_chunks]
        self._corpus_texts = [c["document"] for c in all_chunks]
        self._corpus_metadatas = [c["metadata"] for c in all_chunks]

        tokenized_corpus = [self._tokenize(doc) for doc in self._corpus_texts]
        self._bm25 = BM25Okapi(corpus=tokenized_corpus)
        self._stale = False
        logger.info("BM25 index built with %d documents", len(self._corpus_ids))
        self._save_index()

    def invalidate_index(self) -> None:
        """Mark the BM25 index as stale, forcing a rebuild on next search.

        Call this after any ingestion/deletion against the underlying vector
        store — a count-only staleness check would miss same-count
        replacements (e.g. re-ingesting updated documents with the same
        chunk count).
        """
        self._stale = True
        if self.persist_path and self.persist_path.exists():
            try:
                self.persist_path.unlink()
                logger.info("Removed stale BM25 index file: %s", self.persist_path)
            except Exception:
                logger.debug("Failed to remove stale BM25 index file", exc_info=True)

    def _save_index(self) -> None:
        """Serialize and save the BM25 corpus to disk as JSON.

        Only the raw corpus is persisted (not the BM25 object itself).
        The BM25 index is rebuilt cheaply from the corpus on load, which
        avoids the security risk of ``pickle.load`` and makes the index
        portable across Python and rank_bm25 versions.
        """
        if self.persist_path is None:
            return

        import json

        try:
            import os

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "version": 1,
                "corpus_ids": self._corpus_ids,
                "corpus_texts": self._corpus_texts,
                "corpus_metadatas": self._corpus_metadatas,
            }
            # Use os.open with 0o600 so the corpus (which contains document text)
            # is readable only by the owning process — not world-readable.
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(self.persist_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
            logger.info("Saved BM25 corpus to %s", self.persist_path)
        except Exception:
            logger.exception("Failed to save BM25 corpus to %s", self.persist_path)

    # Maximum size in bytes we are willing to load from disk (500 MB).
    _MAX_INDEX_FILE_BYTES = 500 * 1024 * 1024

    def _load_index(self) -> bool:
        """Load BM25 corpus from disk and rebuild the index. Returns True on success."""
        if self.persist_path is None or not self.persist_path.exists():
            return False

        # Guard against unbounded memory usage from a corrupted/oversized index file.
        try:
            file_size = self.persist_path.stat().st_size
        except OSError:
            return False
        if file_size > self._MAX_INDEX_FILE_BYTES:
            logger.warning(
                "BM25 index file %s is %.0f MB, exceeding the %d MB safety limit — skipping disk load",
                self.persist_path,
                file_size / (1024 * 1024),
                self._MAX_INDEX_FILE_BYTES // (1024 * 1024),
            )
            return False

        import json

        try:
            with open(self.persist_path, encoding="utf-8") as f:
                state = json.load(f)
            self._corpus_ids = state["corpus_ids"]
            self._corpus_texts = state["corpus_texts"]
            self._corpus_metadatas = state["corpus_metadatas"]
            # Rebuild BM25 from the loaded corpus (fast, no pickle risk)
            tokenized = [self._tokenize(doc) for doc in self._corpus_texts]
            if not tokenized:
                self._bm25 = BM25Okapi(corpus=[[""]])
            else:
                self._bm25 = BM25Okapi(corpus=tokenized)
            self._stale = False
            logger.info(
                "Loaded BM25 corpus from %s and rebuilt index (%d docs)",
                self.persist_path,
                len(self._corpus_ids),
            )
            return True
        except Exception:
            logger.warning("Failed to load BM25 corpus from %s, will rebuild", self.persist_path)
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        where: dict[str, str | int | float] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search: BM25 + vector, fused with RRF.

        Args:
            query: The search query string.
            k: Number of final results.
            where: Optional metadata filter for vector search.

        Returns:
            Ranked list of result dicts (id, document, metadata, score).
        """
        # --- BM25 scores ---
        bm25_results = self._bm25_search(query, k)
        bm25_rank = {r["id"]: i for i, r in enumerate(bm25_results)}

        # --- Vector scores ---
        vector_results = self.vector_store.similarity_search(query, k=k, where=where)
        vector_rank = {r["id"]: i for i, r in enumerate(vector_results)}

        # --- RRF fusion ---
        all_ids = set(bm25_rank.keys()) | set(vector_rank.keys())

        rrf_scores: dict[str, float] = {}
        for doc_id in all_ids:
            bm25_r = bm25_rank.get(doc_id, k)  # default to worst rank
            vec_r = vector_rank.get(doc_id, k)
            # Weighted RRF
            score = self.alpha * (1.0 / (self.rrf_k + vec_r + 1)) + (1.0 - self.alpha) * (
                1.0 / (self.rrf_k + bm25_r + 1)
            )
            rrf_scores[doc_id] = score

        # --- Build result list ---
        seen_docs: dict[str, dict[str, Any]] = {}
        for r in bm25_results:
            seen_docs[r["id"]] = r
        for r in vector_results:
            if r["id"] not in seen_docs:
                seen_docs[r["id"]] = r

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]

        results = []
        for doc_id in sorted_ids:
            if doc_id in seen_docs:
                result = dict(seen_docs[doc_id])
                result["score"] = round(rrf_scores[doc_id], 4)
                results.append(result)

        return results

    def _bm25_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Run BM25 keyword search."""
        current_count = self.vector_store.count()

        # Load from disk if current state is memory-cold but not stale
        if self._bm25 is None and not self._stale:
            self._load_index()

        # Rebuild if still missing, stale, or count differs from DB
        if self._bm25 is None or self._stale or len(self._corpus_ids) != current_count:
            # Try loading from disk if memory is clean but disk index matches DB count
            if self._bm25 is None and self._load_index() and len(self._corpus_ids) == current_count:
                pass
            else:
                self.build_index()

        if self._bm25 is None:
            raise RuntimeError("BM25 index unavailable after build attempt")

        if not self._corpus_ids:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "id": self._corpus_ids[idx],
                        "document": self._corpus_texts[idx],
                        "metadata": self._corpus_metadatas[idx],
                        "score": round(float(scores[idx]), 4),
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split on non-alphanumeric boundaries, preserving Unicode letters.

        Handles hyphenated terms (e.g. "GPT-4" → ["gpt", "4"]), punctuation-adjacent
        words, and multilingual Latin-extended characters used in de/es/fr text.
        Empty tokens are filtered out.
        """
        import re

        return [t for t in re.split(r"[^a-zA-Z0-9À-ɏ]+", text.lower()) if t]
