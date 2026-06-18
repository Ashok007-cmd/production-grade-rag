"""Tests for multi-language support, Accept-Language middleware, and language-specific collection routing."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.ingestion.chunker import Chunk
from src.pipeline import RAGPipeline
from src.utils.i18n import _, set_locale


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def test_locale_compilation_exists() -> None:
    """Verify that compiled locales (.mo files) are present for de and es."""
    locale_dir = Path(__file__).parent.parent / "src" / "locale"
    assert (locale_dir / "de" / "LC_MESSAGES" / "messages.mo").exists()
    assert (locale_dir / "es" / "LC_MESSAGES" / "messages.mo").exists()
    assert (locale_dir / "en" / "LC_MESSAGES" / "messages.mo").exists()


def test_i18n_context_translation() -> None:
    """Verify that the translation context sets and translates strings correctly."""
    # Test English (default / fallback)
    token = set_locale("en")
    try:
        assert _("Question must not be empty.") == "Question must not be empty."
    finally:
        from src.utils.i18n import _current_translation
        _current_translation.reset(token)

    # Test German
    token = set_locale("de")
    try:
        assert _("Question must not be empty.") == "Die Frage darf nicht leer sein."
    finally:
        _current_translation.reset(token)

    # Test Spanish
    token = set_locale("es")
    try:
        assert _("Question must not be empty.") == "La pregunta no debe estar vacía."
    finally:
        _current_translation.reset(token)


def test_api_middleware_i18n_translation(api_client: TestClient) -> None:
    """Verify that the FastAPI middleware correctly parses Accept-Language header."""
    # 1. No header or English
    response = api_client.post("/query", json={"question": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question must not be empty."

    # 2. German header
    response = api_client.post("/query", json={"question": ""}, headers={"Accept-Language": "de-DE,de;q=0.9"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Die Frage darf nicht leer sein."

    # 3. Spanish header
    response = api_client.post("/query", json={"question": ""}, headers={"Accept-Language": "es"})
    assert response.status_code == 400
    assert response.json()["detail"] == "La pregunta no debe estar vacía."


def test_multilingual_ingestion_routing(temp_chroma_dir: Path) -> None:
    """Verify that documents are routed to separate collections based on detected language."""
    pipeline = RAGPipeline()
    pipeline.config = pipeline.config.model_copy(update={"chroma_path": temp_chroma_dir})

    # Prepare chunks of different languages
    english_chunk = Chunk(content="This is a simple document written in English.", metadata={"source": "en.txt"}, doc_id="d1")
    german_chunk = Chunk(content="Dies ist ein einfaches Dokument, das auf Deutsch verfasst wurde.", metadata={"source": "de.txt"}, doc_id="d2")
    spanish_chunk = Chunk(content="Este es un documento simple escrito en español.", metadata={"source": "es.txt"}, doc_id="d3")

    # Group by language routing (mirroring pipeline.ingest behavior)
    # Direct write to test vector store creation and routing
    pipeline._get_vector_store("en").add_chunks([english_chunk])
    pipeline._get_vector_store("de").add_chunks([german_chunk])
    pipeline._get_vector_store("es").add_chunks([spanish_chunk])

    assert pipeline._get_vector_store("en").count() == 1
    assert pipeline._get_vector_store("de").count() == 1
    assert pipeline._get_vector_store("es").count() == 1

    # Verify retrieval queries route to their corresponding language stores
    results_en = pipeline._retrieve("English document search", lang="en")
    assert len(results_en) == 1
    assert "English" in results_en[0]["document"]

    results_de = pipeline._retrieve("Deutsches Dokument suchen", lang="de")
    assert len(results_de) == 1
    assert "Deutsch" in results_de[0]["document"]

    results_es = pipeline._retrieve("Documento español buscar", lang="es")
    assert len(results_es) == 1
    assert "español" in results_es[0]["document"]
