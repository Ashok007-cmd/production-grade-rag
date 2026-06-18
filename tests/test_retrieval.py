"""Tests for vector store, hybrid retrieval, and re-ranking."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunker import Chunker
from src.ingestion.loader import Document
from src.retrieval.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_store(temp_chroma_dir: Path) -> VectorStore:
    """Create a temporary VectorStore for testing."""
    return VectorStore(
        collection_name="test_collection",
        persist_path=temp_chroma_dir,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
    )


@pytest.fixture
def indexed_store(vector_store: VectorStore) -> VectorStore:
    """Pre-populate a vector store with sample chunks."""
    chunker = Chunker(chunk_size=200, chunk_overlap=20)
    docs = [
        Document(
            content="Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.",
            metadata={"source": "doc1.txt", "filename": "doc1.txt"},
            doc_id="doc1",
        ),
        Document(
            content="Hybrid search merges BM25 keyword search with vector similarity search.",
            metadata={"source": "doc2.txt", "filename": "doc2.txt"},
            doc_id="doc2",
        ),
        Document(
            content="A cross-encoder re-ranker jointly encodes query and document for precise relevance scoring.",
            metadata={"source": "doc3.txt", "filename": "doc3.txt"},
            doc_id="doc3",
        ),
    ]
    chunks = chunker.chunk_many(docs)
    vector_store.add_chunks(chunks)
    return vector_store


# ---------------------------------------------------------------------------
# Vector Store Tests
# ---------------------------------------------------------------------------


class TestVectorStore:
    """Tests for the VectorStore class."""

    def test_empty_store_count(self, vector_store: VectorStore) -> None:
        assert vector_store.count() == 0

    def test_add_chunks(self, vector_store: VectorStore) -> None:
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        doc = Document(
            content="Test content for embedding.",
            doc_id="test_doc",
        )
        chunks = chunker.chunk(doc)
        count = vector_store.add_chunks(chunks)
        assert count >= 1
        assert vector_store.count() >= 1

    def test_similarity_search_returns_results(self, indexed_store: VectorStore) -> None:
        results = indexed_store.similarity_search("What is RAG?", k=2)
        assert len(results) > 0
        assert results[0]["document"]
        assert results[0]["score"] >= 0

    def test_similarity_search_scored_correctly(self, indexed_store: VectorStore) -> None:
        results = indexed_store.similarity_search("RAG generation retrieval", k=3)
        # Ensure scores are in descending order
        scores = [r["score"] for r in results]
        for i in range(len(scores) - 1):
            # Cosine distance 0 = identical -> score 1.0
            assert scores[i] >= scores[i + 1] or abs(scores[i] - scores[i + 1]) < 0.001

    def test_similarity_search_empty_query(self, indexed_store: VectorStore) -> None:
        results = indexed_store.similarity_search("", k=3)
        # Empty query should still return results (just nearest neighbors)
        assert len(results) > 0

    def test_delete_collection(self, vector_store: VectorStore) -> None:
        vector_store.add_chunks(
            [
                type(
                    "obj",
                    (object,),
                    {
                        "chunk_id": "test_id",
                        "content": "test",
                        "doc_id": "d1",
                        "metadata": {"source": "t.txt"},
                    },
                )()
            ]
        )
        assert vector_store.count() > 0
        vector_store.delete_collection()
        assert vector_store.count() == 0

    def test_get_all_chunks(self, indexed_store: VectorStore) -> None:
        all_chunks = indexed_store.get_all_chunks()
        assert len(all_chunks) >= 1
        for chunk in all_chunks:
            assert "id" in chunk
            assert "document" in chunk
            assert "metadata" in chunk

    def test_add_chunks_batches_large_inserts(
        self, vector_store: VectorStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """add_chunks should split inserts into multiple batches when the
        chunk count exceeds the max batch size."""
        from src.ingestion.chunker import Chunk

        monkeypatch.setattr(vector_store, "_max_batch_size", lambda: 2)

        chunks = [
            Chunk(
                content=f"content {i}",
                metadata={"source": "t.txt"},
                chunk_id=f"id-{i}",
                doc_id="d1",
            )
            for i in range(5)
        ]

        add_calls: list[int] = []
        original_add = vector_store._collection.add

        def counting_add(*, ids, documents, metadatas):
            add_calls.append(len(ids))
            return original_add(ids=ids, documents=documents, metadatas=metadatas)

        monkeypatch.setattr(vector_store._collection, "add", counting_add)

        count = vector_store.add_chunks(chunks)

        assert count == 5
        assert add_calls == [2, 2, 1]
        assert vector_store.count() == 5


# ---------------------------------------------------------------------------
# Hybrid & Reranker Tests (Phase 2)
# ---------------------------------------------------------------------------


class TestHybridRetrieval:
    """Tests for hybrid (BM25 + vector) retrieval."""

    def test_hybrid_search_importable(self) -> None:
        from src.retrieval.hybrid import HybridRetriever

        assert HybridRetriever is not None

    def test_hybrid_search_invalid_alpha(self, indexed_store: VectorStore) -> None:
        from src.retrieval.hybrid import HybridRetriever

        with pytest.raises(ValueError, match="alpha"):
            HybridRetriever(indexed_store, alpha=1.5)

    def test_hybrid_build_index(self, indexed_store: VectorStore) -> None:
        from src.retrieval.hybrid import HybridRetriever

        hybrid = HybridRetriever(indexed_store, alpha=0.6)
        hybrid.build_index()
        assert hybrid._bm25 is not None

    def test_hybrid_search_basic(self, indexed_store: VectorStore) -> None:
        from src.retrieval.hybrid import HybridRetriever

        hybrid = HybridRetriever(indexed_store, alpha=0.6)
        hybrid.build_index()
        results = hybrid.search("RAG generation", k=3)
        assert len(results) > 0
        for r in results:
            assert "score" in r
            assert 0 <= r["score"] <= 1

    def test_hybrid_search_empty(self, indexed_store: VectorStore) -> None:
        from src.retrieval.hybrid import HybridRetriever

        hybrid = HybridRetriever(indexed_store, alpha=0.6)
        hybrid.build_index()
        results = hybrid.search("xyznonexistent12345", k=3)
        # Still ok - can return results
        assert isinstance(results, list)


class TestReranker:
    """Tests for cross-encoder re-ranker."""

    def test_reranker_importable(self) -> None:
        from src.retrieval.reranker import CrossEncoderReranker

        assert CrossEncoderReranker is not None

    def test_reranker_empty_results(self) -> None:
        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", [])
        assert results == []

    def test_reranker_real_predict_mocked(self) -> None:
        """Verify cross-encoder re-ranking sorting and scoring with mock model prediction."""
        from unittest.mock import MagicMock, patch

        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.85, 0.42]

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            results = [
                {"document": "Doc 1", "id": "1"},
                {"document": "Doc 2", "id": "2"},
            ]
            reranked = reranker.rerank("query", results, top_k=2)
            assert len(reranked) == 2
            assert reranked[0]["rerank_score"] == 0.85
            assert reranked[1]["rerank_score"] == 0.42

    def test_reranker_import_error(self) -> None:
        """Verify cross-encoder raises ImportError when sentence-transformers is missing."""
        from unittest.mock import patch

        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                reranker.rerank("query", [{"document": "test"}])

    def test_hybrid_auto_rebuild_on_change(self, indexed_store: VectorStore) -> None:
        """Verify BM25 index automatically rebuilds when database count changes."""
        from src.retrieval.hybrid import HybridRetriever

        hybrid = HybridRetriever(indexed_store, alpha=0.6)
        hybrid.build_index()
        initial_length = len(hybrid._corpus_ids)

        # Ingest a new chunk to change vector store count
        chunker = Chunker(chunk_size=200, chunk_overlap=20)
        indexed_store.add_chunks(
            chunker.chunk(
                Document(
                    content="Completely new text for checking BM25 auto-rebuild functionality.",
                    doc_id="new_doc",
                )
            )
        )

        # Search should trigger auto-rebuild
        hybrid.search("BM25 auto-rebuild", k=3)
        assert len(hybrid._corpus_ids) == initial_length + 1

    def test_hybrid_index_persistence(self, indexed_store: VectorStore, tmp_path: Path) -> None:
        """Verify BM25 index can be saved to and loaded from disk (JSON format)."""
        from unittest.mock import patch

        from src.retrieval.hybrid import HybridRetriever

        persist_file = tmp_path / "bm25.json"

        # 1. Build and save
        hybrid1 = HybridRetriever(indexed_store, alpha=0.6, persist_path=persist_file)
        hybrid1.build_index()
        assert persist_file.exists()
        initial_ids = hybrid1._corpus_ids
        assert len(initial_ids) > 0

        # 2. Load from disk and verify build_index is bypassed
        hybrid2 = HybridRetriever(indexed_store, alpha=0.6, persist_path=persist_file)

        with patch.object(hybrid2, "build_index", wraps=hybrid2.build_index) as mock_build:
            results = hybrid2.search("RAG combinations", k=2)
            assert len(results) > 0
            # Ensure build_index was NOT called because index was loaded from disk
            mock_build.assert_not_called()
            assert hybrid2._corpus_ids == initial_ids
            assert hybrid2._bm25 is not None

    def test_vector_store_http_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify VectorStore initializes chromadb.HttpClient when chroma_host is provided."""
        from unittest.mock import MagicMock

        import chromadb

        mock_http_client = MagicMock()
        monkeypatch.setattr(chromadb, "HttpClient", mock_http_client)

        _ = VectorStore(
            collection_name="test_http_collection",
            chroma_host="localhost",
            chroma_port=8000,
        )

        mock_http_client.assert_called_once()
        called_kwargs = mock_http_client.call_args[1]
        assert called_kwargs["host"] == "localhost"
        assert called_kwargs["port"] == 8000
