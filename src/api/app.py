"""FastAPI application exposing the RAG pipeline as a long-lived HTTP service.

Running this app keeps the embedding model and Chroma client warm in memory,
avoiding the per-CLI-invocation cold start incurred by scripts/*.py.

Run locally with:
    uvicorn src.api.app:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings as _settings
from src.generation.citations import CitationFormatter
from src.pipeline import RAGPipeline
from src.utils.i18n import _

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional API key authentication
# ---------------------------------------------------------------------------
_RAG_API_KEY = os.environ.get("RAG_API_KEY", "").strip()


def _check_api_key(authorization: str | None = Header(None)) -> None:
    """Dependency that enforces Bearer token auth when RAG_API_KEY is set."""
    if not _RAG_API_KEY:
        return  # Auth disabled — no key configured
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization[len("Bearer ") :]
    # Constant-time comparison — a naive `!=` short-circuits on the first
    # mismatched byte, leaking a timing side-channel that lets a remote
    # attacker recover the key byte-by-byte across many requests.
    if not secrets.compare_digest(token, _RAG_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key.")


async def setup_locale(accept_language: str | None = Header(None)):
    import gettext

    from src.utils.i18n import _current_translation

    lang = "en"
    if accept_language:
        parts = [p.split(";")[0].split("-")[0].strip().lower() for p in accept_language.split(",")]
        for p in parts:
            if p in ["de", "es", "en"]:
                lang = p
                break

    try:
        translation = gettext.translation(
            domain="messages",
            localedir=str(Path(__file__).parent.parent / "locale"),
            languages=[lang],
            fallback=True,
        )
    except Exception:
        translation = gettext.NullTranslations()

    token = _current_translation.set(translation)
    try:
        yield
    finally:
        _current_translation.reset(token)


app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
    description="HTTP service layer for the Production-Grade RAG pipeline.",
    dependencies=[Depends(setup_locale), Depends(_check_api_key)],
)


class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Propagate or generate a unique X-Request-ID for every request.

    Enables distributed trace correlation across Langfuse, OTel, and logs.
    The client may supply its own ID; we forward it unchanged, or generate
    a UUID4 if absent.
    """

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


app.add_middleware(_RequestIDMiddleware)

_cors_origins_raw = os.environ.get("RAG_CORS_ORIGINS", "*")
_cors_origins: list[str] = (
    ["*"]
    if _cors_origins_raw.strip() == "*"
    else [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Accept-Language", "Content-Type"],
)


# --- OpenAI Exception Handlers ---
try:
    import openai

    @app.exception_handler(openai.RateLimitError)
    async def openai_rate_limit_handler(request: Request, exc: openai.RateLimitError):
        return Response(
            status_code=429,
            content='{"detail": "Rate limit exceeded on upstream LLM provider API."}',
            media_type="application/json",
        )

    @app.exception_handler(openai.APIConnectionError)
    async def openai_connection_handler(request: Request, exc: openai.APIConnectionError):
        return Response(
            status_code=503,
            content='{"detail": "Failed to connect to upstream LLM provider API."}',
            media_type="application/json",
        )

    @app.exception_handler(openai.APIStatusError)
    async def openai_status_handler(request: Request, exc: openai.APIStatusError):
        status = 502
        if exc.status_code == 429:
            status = 429
        return Response(
            status_code=status,
            content=f'{{"detail": "Upstream LLM provider returned error status: {exc.status_code}."}}',
            media_type="application/json",
        )
except ImportError:
    pass

# --- Anthropic Exception Handlers ---
try:
    import anthropic

    @app.exception_handler(anthropic.RateLimitError)
    async def anthropic_rate_limit_handler(request: Request, exc: anthropic.RateLimitError):
        return Response(
            status_code=429,
            content='{"detail": "Rate limit exceeded on upstream LLM provider API."}',
            media_type="application/json",
        )

    @app.exception_handler(anthropic.APIConnectionError)
    async def anthropic_connection_handler(request: Request, exc: anthropic.APIConnectionError):
        return Response(
            status_code=503,
            content='{"detail": "Failed to connect to upstream LLM provider API."}',
            media_type="application/json",
        )

    @app.exception_handler(anthropic.APIStatusError)
    async def anthropic_status_handler(request: Request, exc: anthropic.APIStatusError):
        status = 502
        if exc.status_code == 429:
            status = 429
        return Response(
            status_code=status,
            content=f'{{"detail": "Upstream LLM provider returned error status: {exc.status_code}."}}',
            media_type="application/json",
        )
except ImportError:
    pass

# Module-level singleton, constructed lazily on first use so that /healthz
# does not force-load the embedding model or Chroma client.
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return the shared RAGPipeline instance, constructing it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def reset_pipeline() -> None:
    """Drop the cached pipeline singleton so it is rebuilt on next access.

    Primarily useful for tests, where fixtures patch configuration (e.g.
    ``settings.chroma_path``) after this module has already been imported.
    """
    global _pipeline
    _pipeline = None


class HealthResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    chunks_in_store: int
    embedding_model: str
    llm_provider: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int


class IngestRequest(BaseModel):
    source: str = Field(..., description="Path to a file or directory to ingest.")
    reset: bool = Field(False, description="If true, clear the vector store before ingesting.")


class IngestResponse(BaseModel):
    chunks_ingested: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    use_hybrid: bool = False
    use_reranker: bool = False


class CitationResponse(BaseModel):
    chunk_id: str
    source: str
    filename: str
    text_snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Lightweight liveness check that does not load heavy models."""
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    """Readiness probe checking database access and eager-loading models."""
    try:
        # 1. Eagerly import heavy dependencies
        import rank_bm25  # noqa: F401
        import sentence_transformers  # noqa: F401

        # 2. Warm up components for all supported languages
        pipeline = get_pipeline()
        for lang in ["en", "de", "es"]:
            _ = pipeline._get_vector_store(lang)
            _ = pipeline._get_hybrid_retriever(lang)
        _ = pipeline._get_reranker()

        return HealthResponse(status="ok")
    except Exception as exc:
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Service not ready.",
        ) from exc


@app.get("/stats", response_model=StatsResponse)
def stats() -> dict[str, Any]:
    """Return pipeline statistics, constructing the pipeline if needed."""
    pipeline = get_pipeline()
    return pipeline.stats()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest documents from a file or directory into the vector store."""
    import asyncio

    # Resolve to an absolute path.
    try:
        source_path = Path(request.source).resolve(strict=False)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc

    # Confine ingest paths to the configured data directory to prevent path-traversal.
    allowed_root = Path(_settings.data_dir).resolve()
    try:
        source_path.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Path must be inside the configured data directory ({allowed_root}).",
        ) from None

    if not source_path.exists():
        raise HTTPException(
            status_code=400,
            detail=_("Source path does not exist: {source_path}").format(
                source_path=request.source
            ),
        )

    pipeline = get_pipeline()

    if request.reset:
        await asyncio.to_thread(pipeline.reset)

    try:
        chunks_ingested = await asyncio.to_thread(pipeline.ingest, source_path)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stats = await asyncio.to_thread(pipeline.stats)
    total_chunks = stats["chunks_in_store"]
    return IngestResponse(chunks_ingested=chunks_ingested, total_chunks=total_chunks)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, response: Response) -> QueryResponse:
    """Answer a question using the RAG pipeline."""
    from src.utils.usage import UsageTracker, request_usage

    tracker = UsageTracker()
    token = request_usage.set(tracker)

    try:
        pipeline = get_pipeline()

        try:
            answer, citations = await pipeline.query_async(
                request.question,
                top_k=request.top_k,
                use_hybrid=request.use_hybrid,
                use_reranker=request.use_reranker,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        citation_responses = [
            CitationResponse(
                chunk_id=c["chunk_id"],
                source=c["source"],
                filename=c["filename"],
                text_snippet=c["text_snippet"],
                score=c["score"],
            )
            for c in CitationFormatter.to_dict(citations)
        ]

        # Populate response headers with token usage metrics
        response.headers["X-RAG-Prompt-Tokens"] = str(tracker.prompt_tokens)
        response.headers["X-RAG-Completion-Tokens"] = str(tracker.completion_tokens)
        response.headers["X-RAG-Total-Tokens"] = str(tracker.total_tokens)
        response.headers["X-RAG-LLM-Latency-Sec"] = f"{tracker.total_latency:.4f}"

        return QueryResponse(answer=answer, citations=citation_responses)
    finally:
        request_usage.reset(token)


# ---------------------------------------------------------------------------
# Streaming endpoint — Server-Sent Events
# ---------------------------------------------------------------------------


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Answer a question and stream tokens via Server-Sent Events (SSE).

    Clients should connect with ``Accept: text/event-stream``.

    Each SSE event is one of:
    - ``data: {"token": "<text>"}``   — a generated text chunk
    - ``data: {"citations": [...]}``   — final citation list (last event before DONE)
    - ``data: [DONE]``                 — stream complete

    Example (curl)::

        curl -N -X POST http://localhost:8000/query/stream \\
             -H 'Content-Type: application/json' \\
             -d '{"question": "What is RAG?", "use_hybrid": true}'
    """
    pipeline = get_pipeline()

    async def _event_stream() -> AsyncGenerator[str, None]:
        try:
            question = request.question.strip()
            if not question:
                yield f"data: {json.dumps({'error': 'Question must not be empty.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            if len(question) > RAGPipeline.MAX_QUESTION_LENGTH:
                yield f"data: {json.dumps({'error': 'Question exceeds maximum length.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            k = request.top_k or _settings.top_k_final

            # Retrieval runs in a thread (blocking I/O to ChromaDB / BM25)
            contexts = await asyncio.to_thread(
                pipeline._retrieve,
                question,
                use_hybrid=request.use_hybrid,
                use_reranker=request.use_reranker,
                k=k,
            )

            if not contexts:
                no_context_msg = (
                    "I could not find any relevant information in the knowledge "
                    "base to answer your question."
                )
                yield f"data: {json.dumps({'token': no_context_msg})}\n\n"
                yield f"data: {json.dumps({'citations': []})}\n\n"
                yield "data: [DONE]\n\n"
                return

            if request.use_reranker:
                contexts = await asyncio.to_thread(
                    pipeline._apply_reranker, question, contexts, top_k=k
                )

            contexts = pipeline._apply_context_budget(contexts)

            # Stream LLM tokens
            async for chunk in pipeline.generator.generate_stream(question, contexts):
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            # Emit citations as final structured event
            citations = pipeline.citation_formatter.build_citations(contexts)
            citation_dicts = CitationFormatter.to_dict(citations)
            yield f"data: {json.dumps({'citations': citation_dicts})}\n\n"

        except ValueError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': 'Internal server error during streaming.'})}\n\n"
            raise exc
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
