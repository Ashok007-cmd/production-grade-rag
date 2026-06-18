"""Main RAG pipeline orchestrating ingestion → retrieval → reranking → generation."""

from __future__ import annotations

import contextvars
import gettext
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

from src.config import settings
from src.generation.citations import Citation, CitationFormatter
from src.generation.generator import Generator
from src.ingestion.chunker import Chunker
from src.ingestion.loader import DocumentLoader
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_SUPPORTED_LANGUAGES: frozenset[str] = frozenset(["en", "de", "es"])


def _detect_language(text: str) -> str:
    """Detect language of text, falling back to 'en' for unsupported languages."""
    from langdetect import LangDetectException, detect

    try:
        lang = detect(text)
    except LangDetectException:
        return "en"
    return lang if lang in _SUPPORTED_LANGUAGES else "en"


def _setup_translation(lang: str) -> contextvars.Token | None:
    """Install a gettext translation for ``lang`` if none is already active.

    Returns a context-var Token for cleanup (or None if the caller need not reset).
    """
    from src.utils.i18n import _current_translation, get_translation

    if isinstance(get_translation(), gettext.GNUTranslations):
        return None
    try:
        translation = gettext.translation(
            domain="messages",
            localedir=str(Path(__file__).parent / "locale"),
            languages=[lang],
            fallback=True,
        )
        return _current_translation.set(translation)
    except Exception:
        return None


class RAGPipeline:
    """End-to-end RAG pipeline: ingest documents, retrieve context, generate answers.

    Supports automatic language routing to separate collections per language,
    and runtime localization of prompts and responses.
    """

    MAX_QUESTION_LENGTH = 2000

    def __init__(self, llm_provider: Literal["openai", "anthropic"] | None = None) -> None:
        self.config = settings

        # Ingestion
        self.loader = DocumentLoader(
            max_file_size_mb=self.config.max_file_size_mb,
            max_pdf_pages=self.config.max_pdf_pages,
        )
        self.chunker = Chunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy="recursive",
        )

        # Dictionary of vector stores and hybrid retrievers, mapped by language
        self._vector_stores: dict[str, VectorStore] = {}
        self._hybrid_retrievers: dict[str, HybridRetriever] = {}

        # Generation
        self.generator = Generator(
            provider=llm_provider or self.config.llm_provider,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        self.citation_formatter = CitationFormatter()

        # Phase 2: Reranker (lazy-loaded)
        self._reranker: CrossEncoderReranker | None = None

    @property
    def vector_store(self) -> VectorStore:
        """The Chroma-backed vector store (defaulting to English for backward compatibility)."""
        return self._get_vector_store("en")

    def _get_vector_store(self, lang: str) -> VectorStore:
        """Construct and retrieve the vector store for a specific language."""
        if lang not in self._vector_stores:
            collection_name = f"rag_docs_{lang}"
            self._vector_stores[lang] = VectorStore(
                collection_name=collection_name,
                persist_path=self.config.chroma_path,
                embedding_model=self.config.embedding_model,
                embedding_dim=self.config.embedding_dim,
                chroma_host=self.config.chroma_host,
                chroma_port=self.config.chroma_port,
            )
        return self._vector_stores[lang]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source: Path | str) -> int:
        """Load, chunk, and index documents from a file or directory.

        Automatically detects document chunk language and routes to the matching collection.

        Args:
            source: Path to a file or directory.

        Returns:
            Number of chunks ingested.
        """
        docs = self.loader.load(source)
        chunks = self.chunker.chunk_many(docs)

        chunks_by_lang: dict[str, list[Any]] = {}
        for chunk in chunks:
            lang = _detect_language(chunk.content)
            chunks_by_lang.setdefault(lang, []).append(chunk)

        total_count = 0
        for lang, lang_chunks in chunks_by_lang.items():
            vs = self._get_vector_store(lang)
            count = vs.add_chunks(lang_chunks)
            total_count += count

            # Invalidate corresponding hybrid index
            if lang in self._hybrid_retrievers:
                self._hybrid_retrievers[lang].invalidate_index()

        logger.info("Ingested %d chunks from %s", total_count, source)
        return total_count

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> tuple[str, list[Citation]]:
        """Answer a question using the RAG pipeline.

        Args:
            question: The user's query string.
            top_k: Number of final context chunks (default: from config).
            use_hybrid: Enable BM25 + vector hybrid search (Phase 2).
            use_reranker: Enable cross-encoder re-ranking (Phase 2).

        Returns:
            Tuple of (answer_text, list_of_citations).

        Raises:
            ValueError: If the question is empty or exceeds the maximum length.
        """
        from src.utils.i18n import _, _current_translation

        lang = _detect_language(question)
        token = _setup_translation(lang)

        try:
            question = question.strip()
            if not question:
                raise ValueError(_("Question must not be empty."))
            if len(question) > self.MAX_QUESTION_LENGTH:
                raise ValueError(
                    _(
                        "Question is too long ({len_question} chars). "
                        "Maximum allowed is {max_len} characters."
                    ).format(len_question=len(question), max_len=self.MAX_QUESTION_LENGTH)
                )

            k = top_k or self.config.top_k_final

            contexts = self._retrieve(question, use_hybrid=use_hybrid, use_reranker=use_reranker, k=k, lang=lang)

            if not contexts:
                logger.warning("No relevant context found for query: %s", question)
                return (
                    _(
                        "I could not find any relevant information in the knowledge base to answer your question."
                    ),
                    [],
                )

            if use_reranker:
                contexts = self._apply_reranker(question, contexts, top_k=k)

            contexts_for_generation = self._apply_context_budget(contexts)
            answer = self.generator.generate(question, contexts_for_generation)
            citations = self.citation_formatter.build_citations(contexts)

            logger.info("Answered query in %d context chunks", len(contexts))
            return answer, citations
        finally:
            if token is not None:
                _current_translation.reset(token)

    async def query_async(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> tuple[str, list[Citation]]:
        """Answer a question using the RAG pipeline asynchronously.

        Args:
            question: The user's query string.
            top_k: Number of final context chunks (default: from config).
            use_hybrid: Enable BM25 + vector hybrid search (Phase 2).
            use_reranker: Enable cross-encoder re-ranking (Phase 2).

        Returns:
            Tuple of (answer_text, list_of_citations).

        Raises:
            ValueError: If the question is empty or exceeds the maximum length.
        """
        import asyncio

        from src.utils.i18n import _, _current_translation

        lang = _detect_language(question)
        token = _setup_translation(lang)

        try:
            question = question.strip()
            if not question:
                raise ValueError(_("Question must not be empty."))
            if len(question) > self.MAX_QUESTION_LENGTH:
                raise ValueError(
                    _(
                        "Question is too long ({len_question} chars). "
                        "Maximum allowed is {max_len} characters."
                    ).format(len_question=len(question), max_len=self.MAX_QUESTION_LENGTH)
                )

            k = top_k or self.config.top_k_final

            contexts = await asyncio.to_thread(
                self._retrieve,
                question,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                k=k,
                lang=lang,
            )

            if not contexts:
                logger.warning("No relevant context found for query: %s", question)
                return (
                    _(
                        "I could not find any relevant information in the knowledge base to answer your question."
                    ),
                    [],
                )

            if use_reranker:
                contexts = await asyncio.to_thread(self._apply_reranker, question, contexts, top_k=k)

            contexts_for_generation = self._apply_context_budget(contexts)
            answer = await self.generator.generate_async(question, contexts_for_generation)
            citations = self.citation_formatter.build_citations(contexts)

            logger.info("Answered query in %d context chunks (async)", len(contexts))
            return answer, citations
        finally:
            if token is not None:
                _current_translation.reset(token)

    def _retrieve(
        self,
        query: str,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        k: int = 5,
        lang: str = "en",
    ) -> list[dict[str, Any]]:
        """Retrieve relevant context chunks."""
        fetch_k = self.config.top_k_retrieval if use_reranker else k

        if use_hybrid:
            return self._get_hybrid_retriever(lang).search(query, k=fetch_k)
        return self._get_vector_store(lang).similarity_search(query, k=fetch_k)

    def _apply_context_budget(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim retrieved contexts so their combined text stays within ``config.max_context_chars``."""
        budget = self.config.max_context_chars
        if budget <= 0:
            return contexts

        result: list[dict[str, Any]] = []
        used = 0
        for ctx in contexts:
            doc = ctx.get("document", "")
            remaining = budget - used
            if remaining <= 0:
                break
            if len(doc) > remaining:
                ctx = {**ctx, "document": doc[:remaining]}
                result.append(ctx)
                break
            result.append(ctx)
            used += len(doc)

        return result

    def _apply_reranker(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank retrieved contexts using cross-encoder."""
        return self._get_reranker().rerank(
            query,
            contexts,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Lazy-loaded components (Phase 2)
    # ------------------------------------------------------------------

    def _get_hybrid_retriever(self, lang: str = "en") -> HybridRetriever:
        if lang not in self._hybrid_retrievers:
            from src.retrieval.hybrid import HybridRetriever

            self._hybrid_retrievers[lang] = HybridRetriever(
                vector_store=self._get_vector_store(lang),
                alpha=self.config.hybrid_alpha,
                rrf_k=self.config.rrf_k,
            )
            self._hybrid_retrievers[lang].build_index()
        return self._hybrid_retrievers[lang]

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            from src.retrieval.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(
                model_name=self.config.reranker_model,
            )
        return self._reranker

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the vector store and all cached components."""
        # Reset default vector store
        self.vector_store.delete_collection()

        # Reset all other initialized language vector stores
        for lang, vs in list(self._vector_stores.items()):
            if lang != "en":
                vs.delete_collection()

        self._vector_stores.clear()
        self._hybrid_retrievers.clear()
        self._reranker = None
        logger.info("Pipeline reset: vector store cleared")

    def stats(self) -> dict[str, Any]:
        """Return pipeline statistics (defaulting to English for backward compatibility)."""
        return {
            "chunks_in_store": self.vector_store.count(),
            "embedding_model": self.config.embedding_model,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }
