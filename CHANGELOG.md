# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-07-02

### Security
- Closed an unauthenticated pre-auth RCE (chromadb CVE PYSEC-2026-311) exposed via `docker-compose.yml` publishing ChromaDB's raw REST API to the host network, bypassing this project's own API auth entirely.
- Fixed a timing side-channel in API key comparison (`secrets.compare_digest` instead of `!=`).
- Stopped `/readyz` from leaking internal exception details to unauthenticated clients.
- Fixed `/healthz` incorrectly requiring auth, which would have caused Docker/Kubernetes health checks to fail and restart the container in a loop whenever `RAG_API_KEY` was enabled.
- Fixed a mypy/numpy stub incompatibility that was silently breaking the CI type-check job on a fresh dependency install.

### Added
- **Prometheus `/metrics` endpoint** — request count and latency histograms by route, independent of the existing OTel/Langfuse pipeline.
- **Query embedding cache** — in-process LRU cache for repeated/paraphrased query embeddings (`RAG_EMBEDDING_QUERY_CACHE_SIZE`), ~10x faster on a cache hit; hit/miss stats surfaced via `/stats`.
- **Async ingestion** — `POST /ingest/async` + `GET /ingest/jobs/{job_id}` for background ingestion with job-ID polling, avoiding client/proxy timeouts on large corpora.
- **Full RAGAS-style evaluation suite** — added `ContextPrecisionScorer` (Average Precision @ k) and `ContextRecallScorer` (reference-answer statement attribution), completing all four standard RAG quality dimensions (faithfulness, answer relevance, context precision, context recall).

### Repository / CI
- Branch protection on `main` with required status checks.
- Dependabot enabled (pip, GitHub Actions, Docker base image) plus GitHub vulnerability alerts and automated security fixes.
- Added `SECURITY.md`.
- Docker publish workflow now cancels superseded in-progress builds on new pushes.

