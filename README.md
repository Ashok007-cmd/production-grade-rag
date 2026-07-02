# Production-Grade RAG Pipeline

[![CI](https://github.com/Ashok007-cmd/production-grade-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashok007-cmd/production-grade-rag/actions/workflows/ci.yml)
[![Evaluate](https://github.com/Ashok007-cmd/production-grade-rag/actions/workflows/evaluate.yml/badge.svg)](https://github.com/Ashok007-cmd/production-grade-rag/actions/workflows/evaluate.yml)
[![Docker](https://github.com/Ashok007-cmd/production-grade-rag/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Ashok007-cmd/production-grade-rag/actions/workflows/docker-publish.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-124%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready **Retrieval-Augmented Generation (RAG)** system built with FastAPI, ChromaDB, and dual LLM support (OpenAI + Anthropic). Demonstrates the full spectrum from a clean vector-search baseline to advanced hybrid retrieval, cross-encoder reranking, OpenTelemetry metrics, Langfuse tracing, and LLM-as-Judge evaluation — all wired together in a deployable, observable, and testable package.

---

## Architecture

```
Documents → [Loader → Chunker] → ChromaDB (dense)
                                 BM25 index  (sparse)
                                      ↓
Query → Language detection → HybridRetriever (RRF fusion)
                           → CrossEncoder Reranker
                                      ↓
                           → Generator (OpenAI / Anthropic)
                                      ↓
                           Answer + Citations + OTel traces
```

### Three Progressive Phases

| Phase | What's added | Key files |
|-------|-------------|-----------|
| **1 — Core RAG** | Document loading, recursive chunking, ChromaDB vector store, OpenAI/Anthropic generation, citations | `src/ingestion/`, `src/retrieval/vector_store.py`, `src/generation/` |
| **2 — Hybrid Search + Reranking** | BM25 keyword index, Reciprocal Rank Fusion (RRF), cross-encoder reranker, per-language collections | `src/retrieval/hybrid.py`, `src/retrieval/reranker.py` |
| **3 — Evaluation + Observability** | LLM-as-Judge faithfulness scoring, golden dataset CI gate, OpenTelemetry metrics, Langfuse tracing, prompt registry | `src/evaluation/`, `src/monitoring/` |

---

## Features

- **Hybrid Search** — BM25 (rank-bm25) + ChromaDB vector similarity fused with Reciprocal Rank Fusion (RRF). Alpha controls the balance; the corpus index persists to disk with 0o600 permissions and a 500 MB safety guard.
- **Query Embedding Cache** — An in-process LRU cache (`RAG_EMBEDDING_QUERY_CACHE_SIZE`, default 256) avoids re-running the embedding model for repeated/paraphrased queries — common in eval runs and demos. Cache hit/miss counts are exposed via `/stats` and `/metrics`.
- **Cross-Encoder Reranking** — `BAAI/bge-reranker-large` (configurable) reorders retrieved chunks for maximum relevance before generation.
- **Multilingual Routing** — `langdetect` identifies the query language; documents are sharded into per-language ChromaDB collections (en/de/es) with language-appropriate prompts via gettext.
- **SSE Token Streaming** — `/query/stream` endpoint delivers tokens over Server-Sent Events as they are generated; compatible with any EventSource client.
- **Async Ingestion** — `POST /ingest/async` enqueues ingestion as a background job and returns a `job_id` immediately (202), avoiding client/proxy timeouts on large corpora; poll `GET /ingest/jobs/{job_id}` for status. The server stays fully responsive to other requests while a job runs.
- **Dual LLM Backend** — Switch between OpenAI and Anthropic at runtime via `RAG_LLM_PROVIDER`. Async streaming implemented for both providers.
- **Optional API Key Auth** — Set `RAG_API_KEY` to require `Authorization: Bearer <key>` on all requests. Omit the variable to run without auth (development default).
- **OpenTelemetry Metrics** — Latency histograms, query counters, cost tracking, and error rates exported via OTLP. Falls back gracefully when the OTel backend is absent.
- **Langfuse Tracing** — End-to-end span capture for retrieve → rerank → generate stages with generation cost attribution. Circuit breaker prevents telemetry failures from impacting query latency.
- **Prompt Registry** — SHA-256 hash-based change detection for prompt versioning across deployments.
- **LLM-as-Judge Evaluation** — Faithfulness + answer-relevance scoring via `gpt-4o-mini` against a golden JSONL dataset, with CI quality gate (`--fail-on-threshold`).
- **X-Request-ID Correlation** — Middleware echoes or generates a `X-Request-ID` header on every response for distributed trace correlation.
- **Context Budget Management** — Configurable `max_context_chars` trims retrieved chunks to fit within the LLM context window without truncating mid-sentence.
- **Security Hardening** — Path-traversal guards on `/ingest`, BM25 index serialized as JSON (not pickle), 0o600 file permissions on data files, `re.escape()` on pricing pattern matching to prevent ReDoS.
- **Docker Multi-Stage Build** — `deps → model-cache → runtime` stages; non-root user, no model download at container start.
- **124 Tests, 3 CI Workflows** — Unit + integration tests with coverage, automatic quality gate evaluation, and GHCR Docker publish on release.

---

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key (or Anthropic key if using `RAG_LLM_PROVIDER=anthropic`)

### Installation

```bash
git clone https://github.com/Ashok007-cmd/production-grade-rag.git
cd production-grade-rag

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### Ingest Documents

```bash
# Local CLI (no server needed):
python scripts/ingest.py --source data/sample_docs

# With a running API server the CLI auto-routes to it:
python scripts/ingest.py --source path/to/your/docs --reset
```

### Query

```bash
# Basic vector search:
python scripts/query.py --question "What is Retrieval-Augmented Generation?"

# Phase 2 — hybrid + reranker:
python scripts/query.py --question "How does RRF scoring work?" --hybrid --reranker

# Stream tokens over SSE (requires running server):
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain hybrid search"}'
```

### Run the API Server

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/healthz` | Liveness probe — returns `{"status": "ok"}` |
| `GET` | `/readyz` | Readiness probe — warms embedding model |
| `GET` | `/stats` | Pipeline statistics (chunks, model, config) |
| `GET` | `/metrics` | Prometheus scrape endpoint — request count + latency histograms by route |
| `POST` | `/ingest` | Ingest documents from a server-accessible path (blocks until done) |
| `POST` | `/ingest/async` | Enqueue ingestion as a background job; returns `job_id` immediately (202) |
| `GET` | `/ingest/jobs/{job_id}` | Poll status of an async ingestion job |
| `POST` | `/query` | Synchronous Q&A — returns answer + citations |
| `POST` | `/query/stream` | **SSE streaming** — streams tokens then citations |

### POST /query

```json
{
  "question": "What is hybrid search?",
  "top_k": 5,
  "use_hybrid": true,
  "use_reranker": true
}
```

Response:
```json
{
  "answer": "Hybrid search combines...",
  "citations": [
    {"chunk_id": "abc123", "source": "doc.pdf", "filename": "doc.pdf",
     "text_snippet": "...", "score": 0.842}
  ]
}
```

### POST /query/stream

Same request body as `/query`. Response is a stream of Server-Sent Events:

```
data: {"token": "Hybrid"}
data: {"token": " search"}
...
data: {"citations": [...]}
data: [DONE]
```

### Request Headers

| Header | Description |
|--------|-------------|
| `Authorization: Bearer <key>` | Required when `RAG_API_KEY` is set |
| `X-Request-ID` | Optional correlation ID; echoed in response (generated if absent) |

---

## Configuration

All settings use the `RAG_` prefix and can be set via `.env` or environment variables.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_LLM_PROVIDER` | `openai` | LLM backend: `openai` or `anthropic` |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Model name for the selected provider |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `RAG_DATA_DIR` | `data` | Root directory for data files |
| `RAG_CHROMA_PATH` | `data/chroma_db` | ChromaDB persistence directory |
| `RAG_CHUNK_SIZE` | `800` | Target chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `RAG_TOP_K_RETRIEVAL` | `20` | Candidates fetched before reranking |
| `RAG_TOP_K_FINAL` | `5` | Final context chunks passed to the LLM |
| `RAG_HYBRID_ALPHA` | `0.6` | RRF weight: 1.0 = pure vector, 0.0 = pure BM25 |
| `RAG_EMBEDDING_QUERY_CACHE_SIZE` | `256` | LRU cache size for repeated query embeddings (`0` disables) |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RAG_FAITHFULNESS_THRESHOLD` | `0.7` | Minimum faithfulness score for CI gate |
| `RAG_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `RAG_API_KEY` | *(unset)* | Bearer token for API auth (disabled when unset) |
| `RAG_LOG_LEVEL` | `INFO` | Logging level |

### Monitoring Settings (`MONITOR_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITOR_ENABLED` | `true` | Enable the monitoring subsystem |
| `MONITOR_LANGFUSE_SECRET_KEY` | — | Langfuse secret key (`sk-lf-…`) |
| `MONITOR_LANGFUSE_PUBLIC_KEY` | — | Langfuse public key (`pk-lf-…`) |
| `MONITOR_LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server URL |
| `MONITOR_OTEL_SERVICE_NAME` | `rag-pipeline` | OTel service name |
| `MONITOR_OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTLP receiver endpoint |
| `MONITOR_CIRCUIT_BREAKER_THRESHOLD` | `3` | Failures before circuit opens |
| `MONITOR_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `30.0` | Seconds before half-open retry |

---

## Monitoring & Observability

### OpenTelemetry Metrics

When `MONITOR_ENABLED=true` and an OTLP endpoint is reachable, the pipeline exports:

- **`rag.query.latency`** — histogram of end-to-end query latency per pipeline step
- **`rag.query.count`** — counter of total queries processed
- **`rag.query.errors`** — counter of errors by step and type
- **`rag.query.cost`** — running cost in USD per query

A JSON summary (P50/P95 latency, avg cost, error count) is exported via `MetricsCollector.export_summary()` and written with 0o600 permissions.

### Langfuse Tracing

With Langfuse credentials set, each query creates a trace with spans for:

- `retrieve` — vector/hybrid search timing
- `rerank` — cross-encoder reranking timing
- `generate` — LLM generation with model, token counts, and cost

A circuit breaker (`MONITOR_CIRCUIT_BREAKER_THRESHOLD`) isolates Langfuse failures so a tracing outage never blocks query responses.

### Using the Monitored Pipeline

```python
from src.monitoring import MetricsCollector, MonitoredRAGPipeline, Tracer
from src.pipeline import RAGPipeline

pipeline  = RAGPipeline()
tracer    = Tracer(enabled=True)
metrics   = MetricsCollector(enabled=True)
monitored = MonitoredRAGPipeline(pipeline, tracer=tracer, metrics=metrics)

answer, citations = monitored.query("What is RAG?", use_hybrid=True)
metrics.export_summary("monitoring-summary.json")
```

### Prompt Registry

```python
from src.monitoring.prompts import PromptRegistry

registry    = PromptRegistry()
prompt_hash = registry.register("system_v1", "You are a helpful assistant…")
changed     = registry.detect_change("system_v1", new_prompt_text)
```

---

## Docker

### Build

```bash
docker build -t production-rag:latest .
```

### Run

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e RAG_LLM_PROVIDER=openai \
  -v $(pwd)/data:/app/data \
  production-rag:latest
```

The image is published to GitHub Container Registry on every tagged release:

```bash
docker pull ghcr.io/ashok007-cmd/production-rag:latest
```

---

## Why Not LangChain?

This project builds retrieval, fusion, reranking, and generation directly against `chromadb`, `rank-bm25`, and the OpenAI/Anthropic SDKs instead of a framework. The tradeoffs, deliberately:

- **Every retrieval decision is explicit and testable.** RRF fusion (`src/retrieval/hybrid.py`) and context-budget trimming (`src/pipeline.py`) are plain functions with unit tests — not framework internals to work around when they don't fit a use case.
- **Fewer version-compatibility surprises.** Framework abstraction layers add a dependency-resolution axis (framework version × provider SDK version × vector-store client version) that has historically been a common source of breakage in LangChain-based projects.
- **Debuggability.** A stack trace from `pipeline.query()` points directly at ingestion, retrieval, or generation code — not through several layers of chain/agent abstraction.
- **When a framework *would* make sense:** rapid prototyping across many retrieval strategies, or when the team needs pre-built integrations for dozens of vector stores/tools out of the box. For a single, well-understood production RAG stack, the direct approach costs a small amount of boilerplate for a large amount of control and interview-defensible design decisions.

## Scaling to Production

Notes on what changes as load grows, for anyone evaluating this as a production starting point:

- **Horizontal scaling:** The FastAPI service is stateless aside from the lazily-constructed `RAGPipeline` singleton (`src/api/app.py`); running multiple replicas behind a load balancer works as-is. The one shared-state caveat is the in-process BM25 index cache (`HybridRetriever`) — each replica rebuilds its own copy from the persisted JSON index on first use.
- **ChromaDB clustering:** The current setup uses a single ChromaDB instance (embedded or standalone via `docker-compose.yml`). At higher scale, Chroma's distributed mode or a managed vector DB (e.g. sharded by language collection, which this project already partitions by) is the natural next step — the per-language collection design in `src/pipeline.py` maps cleanly onto shard boundaries.
- **Caching:** Repeated/paraphrased queries currently re-embed and re-retrieve from scratch every time. An embedding-result cache (see the improvement roadmap in `ANALYSIS_REPORT.md`) is the highest-leverage addition before adding more compute.
- **Async ingestion:** Large ingestion jobs currently block the request for their full duration (see `ANALYSIS_REPORT.md` roadmap item D — job-ID + polling pattern) — needed before ingesting corpora large enough to risk proxy timeouts.
- **Observability is already there:** OTel metrics + Langfuse tracing + circuit breakers mean a horizontally-scaled deployment is debuggable from day one — this is usually the last thing added to a scaling project, not the first, and it's already built in here.

---

## Evaluation

The evaluation suite uses **LLM-as-Judge** scoring to measure faithfulness and answer relevance against a golden dataset.

```bash
# Create a sample golden dataset:
python scripts/evaluate.py --create-sample-dataset

# Run evaluation (Phase 2 features + CI exit code):
python scripts/evaluate.py --hybrid --reranker --fail-on-threshold --export-ci-summary
```

The `evaluate.yml` GitHub Actions workflow runs this gate on every push to `main` and posts results as a PR comment.

---

## Testing

```bash
# Full suite (124 tests):
python -m pytest tests/ -v

# With coverage report:
python -m pytest tests/ --cov=src --cov-report=term-missing

# Single module:
python -m pytest tests/test_monitoring.py -v
```

### Test Coverage

| Module | Coverage focus |
|--------|---------------|
| `test_pipeline.py` | End-to-end query/ingest flows |
| `test_retrieval.py` | Vector store, hybrid retriever, RRF scoring |
| `test_generation.py` | Generator, streaming, citation formatting |
| `test_evaluation.py` | LLM-as-Judge scorer, golden dataset, CI summary |
| `test_monitoring.py` | Tracer spans, MetricsCollector, circuit breaker, guardrails |
| `test_api.py` | FastAPI endpoints, auth, SSE streaming |
| `test_chunker.py` | Recursive/fixed/sentence chunking strategies |

---

## Project Structure

```
production-grade-rag/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI app, auth, SSE streaming, middleware
│   ├── ingestion/
│   │   ├── loader.py           # PDF, TXT, Markdown document loading
│   │   └── chunker.py          # Recursive / fixed / sentence chunking
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB wrapper with embedding
│   │   ├── hybrid.py           # BM25 + RRF hybrid retriever
│   │   └── reranker.py         # Cross-encoder reranker
│   ├── generation/
│   │   ├── generator.py        # OpenAI / Anthropic generation + streaming
│   │   └── citations.py        # Citation building and formatting
│   ├── evaluation/
│   │   ├── runner.py           # LLM-as-Judge evaluation runner
│   │   ├── scorer.py           # Faithfulness + relevance scorers
│   │   └── dataset.py          # Golden JSONL dataset management
│   ├── monitoring/
│   │   ├── config.py           # MonitoringSettings (MONITOR_* env vars)
│   │   ├── metrics.py          # OTel MetricsCollector
│   │   ├── tracing.py          # Langfuse Tracer with thread-local spans
│   │   ├── extensions.py       # CircuitBreaker, GuardrailExtension, OTelMetricsExtension
│   │   ├── prompts.py          # SHA-256 prompt registry
│   │   └── wrappers.py         # MonitoredRAGPipeline instrumentation wrapper
│   ├── pipeline.py             # RAGPipeline orchestration + async support
│   ├── config.py               # RAGSettings (RAG_* env vars, pydantic-settings)
│   └── utils/
│       └── i18n.py             # gettext internationalization helpers
├── tests/                      # 124 pytest tests
├── scripts/
│   ├── ingest.py               # CLI document ingestion
│   ├── query.py                # CLI query (auto-routes to API if running)
│   └── evaluate.py             # CLI evaluation runner
├── data/
│   └── sample_docs/            # Sample documents for quick-start
├── .github/workflows/
│   ├── ci.yml                  # Lint, type-check, test (Python 3.11 + 3.12)
│   ├── evaluate.yml            # RAG faithfulness quality gate
│   └── docker-publish.yml      # GHCR publish on release / main push
├── Dockerfile                  # Multi-stage: deps → model-cache → runtime
├── .env.example                # Template for all required env vars
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Dev/test dependencies
└── setup.py                    # Package setup
```

---

## Security

- **API authentication** — `RAG_API_KEY` enables Bearer token auth on all endpoints, checked with a constant-time comparison (`secrets.compare_digest`) to prevent timing-based key recovery.
- **Path traversal protection** — `/ingest` resolves symlinks and validates the source path is contained within the configured data directory before file access.
- **No pickle** — BM25 corpus persists as JSON; no `pickle.load()` anywhere in the codebase.
- **File permissions** — Sensitive data files (BM25 index, metrics export) are written with `os.open(..., 0o600)`.
- **ReDoS prevention** — Pricing key patterns use `re.escape()` before regex compilation.
- **OOM guard** — BM25 index loader rejects files exceeding 500 MB.
- **Secret hygiene** — No partial key logging; keys validated structurally, not compared in logs.
- **No internal error leakage** — Readiness/health failures are logged server-side; clients receive a generic status message.
- **CORS** — `RAG_CORS_ORIGINS` restricts cross-origin access; defaults to `*` for local development only.
- **Deployment-topology hardening** — `docker-compose.yml` does not publish ChromaDB's own REST API to the host; it's reachable only over the internal Docker network by the API service, which prevents bypassing this project's auth layer by hitting Chroma directly.
- **Dependency scanning** — Dependencies are periodically checked with `pip-audit`; see `ANALYSIS_REPORT.md` for the full audit trail and CVE findings.

To report a security vulnerability, please open a private advisory via GitHub Security.

---

## Skills Demonstrated

This project showcases production AI/ML engineering across the full stack:

| Area | Demonstrated by |
|------|----------------|
| **LLM integration** | Dual-provider (OpenAI + Anthropic), async streaming, token tracking |
| **Information retrieval** | BM25, dense vector search, RRF fusion, cross-encoder reranking |
| **API design** | FastAPI, SSE streaming, auth middleware, readiness probes |
| **Observability** | OpenTelemetry metrics, Langfuse distributed tracing, circuit breaker |
| **LLM evaluation** | LLM-as-Judge, faithfulness + relevance metrics, golden dataset CI gate |
| **Security engineering** | Auth, path traversal defense, safe file I/O, no pickle |
| **Testing** | 124 tests, async patterns, mocking strategy, coverage reporting |
| **DevOps / MLOps** | Docker multi-stage build, GitHub Actions CI/CD, GHCR publish |
| **Multilingual NLP** | Language detection, per-language vector collections, i18n prompts |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
