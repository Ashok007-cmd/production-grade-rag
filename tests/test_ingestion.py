"""Tests for document loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunker import Chunk, Chunker
from src.ingestion.loader import Document, DocumentLoader


class TestDocumentLoader:
    """Tests for the DocumentLoader class."""

    def test_load_text_file(self, sample_docs_dir: Path) -> None:
        loader = DocumentLoader()
        file_path = sample_docs_dir / "doc1.txt"
        docs = loader.load(file_path)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "RAG" in docs[0].content
        assert docs[0].metadata["filename"] == "doc1.txt"
        assert docs[0].metadata["filetype"] == ".txt"

    def test_load_directory(self, sample_docs_dir: Path) -> None:
        loader = DocumentLoader()
        docs = loader.load(sample_docs_dir)
        assert len(docs) == 2

    def test_load_nonexistent_path(self) -> None:
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        """Files with unsupported extensions should be skipped."""
        (tmp_path / "image.png").write_text("not-a-text")
        loader = DocumentLoader()
        docs = loader.load(tmp_path)
        assert len(docs) == 0

    def test_metadata_includes_source(self, sample_docs_dir: Path) -> None:
        loader = DocumentLoader()
        docs = loader.load(sample_docs_dir)
        for doc in docs:
            assert "source" in doc.metadata
            assert "filename" in doc.metadata
            assert "filetype" in doc.metadata


class TestChunker:
    """Tests for the Chunker class."""

    def test_recursive_chunking(self, sample_document: Document, chunker: Chunker) -> None:
        chunks = chunker.chunk(sample_document)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.content
            assert chunk.chunk_id
            assert chunk.doc_id == "test_doc_1"

    def test_chunk_metadata_propagation(self, sample_document: Document, chunker: Chunker) -> None:
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"

    def test_chunk_size_respected(self, sample_document: Document) -> None:
        """Each chunk should not exceed chunk_size (plus small tolerance)."""
        chunker = Chunker(chunk_size=100, chunk_overlap=10, strategy="fixed")
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert len(chunk.content) <= 110  # chunk_size + overlap

    def test_fixed_strategy_with_overlap(self) -> None:
        chunker = Chunker(chunk_size=50, chunk_overlap=10, strategy="fixed")
        doc = Document(content="A" * 200, doc_id="overlap_test")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 4  # 200/50 = 4 minimum

    def test_sentence_strategy(self, sample_document: Document) -> None:
        chunker = Chunker(chunk_size=500, chunk_overlap=50, strategy="sentence")
        chunks = chunker.chunk(sample_document)
        assert len(chunks) >= 1

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            Chunker(strategy="invalid")

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            Chunker(chunk_size=100, chunk_overlap=200)

    def test_empty_document(self) -> None:
        chunker = Chunker()
        doc = Document(content="", doc_id="empty")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

    def test_chunk_many(self, sample_document: Document) -> None:
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        doc2 = Document(
            content="Second document for testing.",
            metadata={"source": "test2.txt"},
            doc_id="test_doc_2",
        )
        chunks = chunker.chunk_many([sample_document, doc2])
        assert len(chunks) >= 2

    def test_deterministic_chunk_ids(self, sample_document: Document, chunker: Chunker) -> None:
        """Same content should produce same chunk IDs."""
        chunks1 = chunker.chunk(sample_document)
        chunks2 = chunker.chunk(sample_document)
        for c1, c2 in zip(chunks1, chunks2, strict=False):
            assert c1.chunk_id == c2.chunk_id

    def test_loader_iter_docs(self, sample_docs_dir: Path) -> None:
        """Verify DocumentLoader.iter_docs generator yields documents correctly."""
        loader = DocumentLoader()
        docs_iter = loader.iter_docs(sample_docs_dir)
        docs = list(docs_iter)
        assert len(docs) == 2

    def test_loader_directory_load_error_logged(self, sample_docs_dir: Path) -> None:
        """Verify loader catches and logs exceptions when loading individual directory files."""
        from unittest.mock import patch

        loader = DocumentLoader()
        with patch.object(loader, "_load_file", side_effect=Exception("mocked load failure")):
            docs = loader.load(sample_docs_dir)
            assert len(docs) == 0

    def test_load_pdf_mocked(self, tmp_path: Path) -> None:
        """Verify PDF loader successfully parses pages and metadata using a mocked PdfReader."""
        from unittest.mock import MagicMock, patch

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is text extracted from PDF."
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            loader = DocumentLoader()
            doc = loader._load_pdf(pdf_file)
            assert doc.content == "This is text extracted from PDF."
            assert doc.metadata["filetype"] == ".pdf"
            assert doc.metadata["pages"] == 1
