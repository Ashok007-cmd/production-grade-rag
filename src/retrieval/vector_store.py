"""ChromaDB vector store wrapper for embedding storage and similarity search."""

from __future__ import annotations

import logging
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.chunker import Chunk
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# Module-level cache so all VectorStore instances using the same model share one copy.
_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}


class VectorStore:
    """A ChromaDB-backed vector store for dense retrieval.

    Manages embedding computation and similarity search over document chunks.
    """

    # Fallback used if the Chroma client doesn't expose get_max_batch_size().
    DEFAULT_MAX_BATCH_SIZE = 2000

    def __init__(
        self,
        collection_name: str = "rag_docs",
        persist_path: Path | str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        chroma_host: str | None = None,
        chroma_port: int | None = None,
        embedding_query_cache_size: int = 256,
    ) -> None:
        self.collection_name = collection_name
        self.persist_path = Path(persist_path) if persist_path else Path("data/chroma_db")
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port

        # Lazy-load embedding function
        self._embedding_fn = _ChromaEmbeddingFunction(
            model_name=embedding_model, query_cache_size=embedding_query_cache_size
        )

        # Initialise ChromaDB client
        if self.chroma_host:
            _host: str = self.chroma_host  # narrow str | None → str for the closure

            def _init_client():
                client = chromadb.HttpClient(
                    host=_host,
                    port=self.chroma_port or 8000,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                client.heartbeat()
                return client

            self._client = retry_with_backoff(_init_client, retries=5, backoff_in_seconds=1.0)
        else:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        def _get_or_create():
            return self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        self._collection = retry_with_backoff(_get_or_create, retries=5, backoff_in_seconds=1.0)
        logger.info(
            "VectorStore initialised: collection=%s, path=%s",
            collection_name,
            self.persist_path,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the vector store. Returns count of added chunks.

        Inserts are split into batches bounded by the Chroma client's
        ``max_batch_size`` (when available) to avoid
        ``InvalidArgumentError`` on large ingestions.
        """
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id or str(uuid.uuid4()))
            documents.append(chunk.content)
            metadatas.append({k: str(v) for k, v in chunk.metadata.items()})

        batch_size = self._max_batch_size()

        # Chroma handles embeddings internally via the embedding function
        for start in range(0, len(ids), batch_size):
            end = start + batch_size

            def _add_batch(_s: int = start, _e: int = end) -> None:
                self._collection.add(
                    ids=ids[_s:_e],
                    documents=documents[_s:_e],
                    metadatas=metadatas[_s:_e],
                )

            retry_with_backoff(_add_batch, retries=3, backoff_in_seconds=0.5)
        logger.info("Added %d chunks to collection '%s'", len(ids), self.collection_name)
        return len(ids)

    def _max_batch_size(self) -> int:
        """Return the maximum number of records Chroma accepts per add() call."""
        get_max = getattr(self._client, "get_max_batch_size", None)
        if callable(get_max):
            try:
                return int(get_max())
            except Exception:
                logger.debug("get_max_batch_size() failed, using default", exc_info=True)
        return self.DEFAULT_MAX_BATCH_SIZE

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return retry_with_backoff(self._collection.count, retries=3, backoff_in_seconds=0.5)

    def embedding_cache_stats(self) -> dict[str, int]:
        """Return query-embedding cache hit/miss/size counters for this store."""
        return self._embedding_fn.cache_stats()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str | int | float] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k chunks most similar to the query.

        Returns a list of dicts with keys: id, document, metadata, distance.
        """

        def _search():
            return self._collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        results = retry_with_backoff(_search, retries=3, backoff_in_seconds=0.5)

        return self._format_results(results)

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """Retrieve all chunks (used for hybrid index building)."""

        def _get():
            return self._collection.get(include=["documents", "metadatas"])

        results = retry_with_backoff(_get, retries=3, backoff_in_seconds=0.5)
        return self._format_get_results(results)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def delete_collection(self) -> None:
        """Delete the entire collection."""

        def _delete():
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
            )

        retry_with_backoff(_delete, retries=3, backoff_in_seconds=0.5)
        logger.info("Deleted and recreated collection '%s'", self.collection_name)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove specific chunks by ID."""

        def _delete_ids():
            self._collection.delete(ids=chunk_ids)

        retry_with_backoff(_delete_ids, retries=3, backoff_in_seconds=0.5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results: dict[str, Any]) -> list[dict[str, Any]]:
        """Format ChromaDB query results into a clean list of dicts."""
        formatted: list[dict[str, Any]] = []
        ids_batch = results.get("ids", [[]])[0]
        docs_batch = results.get("documents", [[]])[0]
        metas_batch = results.get("metadatas", [[]])[0]
        dists_batch = results.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids_batch):
            formatted.append(
                {
                    "id": doc_id,
                    "document": docs_batch[idx] if docs_batch else "",
                    "metadata": metas_batch[idx] if metas_batch else {},
                    "score": 1.0 - (dists_batch[idx] if dists_batch else 0.0),
                }
            )
        return formatted

    @staticmethod
    def _format_get_results(results: dict[str, Any]) -> list[dict[str, Any]]:
        """Format ChromaDB get results."""
        formatted: list[dict[str, Any]] = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for i, doc_id in enumerate(ids):
            formatted.append(
                {
                    "id": doc_id,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": 1.0,
                }
            )
        return formatted


class _ChromaEmbeddingFunction:
    """Adapter so ChromaDB can use sentence-transformers models.

    Instances sharing the same model_name reuse the same underlying
    SentenceTransformer object via _EMBEDDING_MODEL_CACHE, so multiple
    VectorStore instances (e.g. per language) don't duplicate model memory.
    """

    def __init__(self, model_name: str, query_cache_size: int = 256) -> None:
        self.model_name = model_name
        self._model = None
        self._query_cache_size = query_cache_size
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    # ChromaDB >= 0.6 calls these methods to detect the embedding function's
    # capabilities and bypass the "legacy" serialization code path.
    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def default_space(self) -> str:
        return "cosine"

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> _ChromaEmbeddingFunction:
        return cls(model_name=config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))

    @staticmethod
    def name() -> str:
        return "sentence-transformers"

    @staticmethod
    def is_legacy() -> bool:
        return False

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        if self.model_name in _EMBEDDING_MODEL_CACHE:
            self._model = _EMBEDDING_MODEL_CACHE[self.model_name]
            logger.debug("Reusing cached embedding model: %s", self.model_name)
            return

        try:
            import torch

            torch.set_num_threads(1)
            logger.info("Constraining PyTorch to 1 CPU thread.")
        except ImportError:
            pass

        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        model = SentenceTransformer(self.model_name)
        _EMBEDDING_MODEL_CACHE[self.model_name] = model
        self._model = model

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed queries for retrieval (ChromaDB >= 0.5).

        Repeated or paraphrased queries (common across eval runs and demos)
        hit an in-process LRU cache instead of re-running the embedding
        model. Document embedding (``embed_document``) intentionally does
        not use this cache — corpus text is rarely repeated and caching it
        would grow unbounded with ingestion volume.
        """
        if self._query_cache_size <= 0:
            return self._encode(input)

        results: list[list[float] | None] = [None] * len(input)
        misses: list[tuple[int, str]] = []

        for i, text in enumerate(input):
            cached = self._query_cache.get(text)
            if cached is not None:
                self._query_cache.move_to_end(text)
                self._cache_hits += 1
                results[i] = cached
            else:
                misses.append((i, text))

        if misses:
            self._cache_misses += len(misses)
            embedded = self._encode([text for _, text in misses])
            for (i, text), vector in zip(misses, embedded, strict=True):
                results[i] = vector
                self._query_cache[text] = vector
                self._query_cache.move_to_end(text)
                if len(self._query_cache) > self._query_cache_size:
                    self._query_cache.popitem(last=False)

        return results  # type: ignore[return-value]  # every slot filled above

    def cache_stats(self) -> dict[str, int]:
        """Return query-embedding cache hit/miss/current-size counters."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._query_cache),
        }

    def embed_document(self, input: list[str]) -> list[list[float]]:
        """Embed documents for indexing (ChromaDB >= 0.5)."""
        return self._encode(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Fallback for older ChromaDB versions that call the object directly."""
        return self._encode(input)

    def _encode(self, input: list[str]) -> list[list[float]]:
        """Core encoding logic shared by all entry points."""
        self._lazy_load()
        if self._model is None:
            raise RuntimeError(f"Embedding model '{self.model_name}' failed to load")
        embeddings = self._model.encode(input, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
