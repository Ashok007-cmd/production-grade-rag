"""Shared test fixtures for RAG pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.ingestion.chunker import Chunker
from src.ingestion.loader import Document


@pytest.fixture(autouse=True)
def isolate_chroma_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force all tests to use an isolated ChromaDB directory, avoiding locks.

    Also sets data_dir to tmp_path so the API path-confinement check passes
    for test fixtures that create documents under tmp_path.
    """
    from src.config import settings

    monkeypatch.setattr(settings, "chroma_path", tmp_path / "test_chroma_rag")
    monkeypatch.setattr(settings, "data_dir", tmp_path)


@pytest.fixture(autouse=True)
def mock_embedding_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the embedding function to return dummy vectors, avoiding loading SentenceTransformer."""
    import hashlib

    from src.retrieval.vector_store import _ChromaEmbeddingFunction

    def dummy_encode(self, input_texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input_texts:
            # Deterministic vector based on text hash
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vector = []
            for i in range(384):
                byte_val = h[i % len(h)]
                val = ((byte_val * (i + 1)) % 256) / 256.0
                vector.append(val)
            embeddings.append(vector)
        return embeddings

    monkeypatch.setattr(_ChromaEmbeddingFunction, "_encode", dummy_encode)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Return a test configuration with small values for fast tests."""
    return Settings(
        chunk_size=200,
        chunk_overlap=20,
        chroma_path=tmp_path / "test_chroma_rag",
        top_k_final=3,
        top_k_retrieval=5,
    )


@pytest.fixture
def sample_text() -> str:
    """Return a multi-paragraph sample text for chunking tests."""
    return (
        "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
        "information retrieval with text generation.\n\n"
        "It retrieves relevant document chunks from a knowledge base and provides "
        "them as context to a large language model.\n\n"
        "This enables the model to generate answers grounded in factual information "
        "rather than relying solely on its training data.\n\n"
        "RAG is widely used in enterprise applications such as customer support "
        "chatbots and internal knowledge base assistants."
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    """Return a Document fixture."""
    return Document(
        content=sample_text,
        metadata={"source": "test.txt", "filename": "test.txt", "filetype": ".txt"},
        doc_id="test_doc_1",
    )


@pytest.fixture
def chunker() -> Chunker:
    """Return a chunker with small chunk size for testing."""
    return Chunker(chunk_size=200, chunk_overlap=20, strategy="recursive")


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for ChromaDB storage."""
    return tmp_path / "chroma_test"


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample text files for ingestion tests."""
    docs_dir = tmp_path / "sample_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "doc1.txt").write_text(
        "RAG combines retrieval with generation for accurate answers."
    )
    (docs_dir / "doc2.txt").write_text(
        "Hybrid search merges BM25 keyword search with vector similarity using RRF."
    )
    return docs_dir
