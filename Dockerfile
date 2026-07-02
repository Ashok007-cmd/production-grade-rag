# syntax=docker/dockerfile:1

# ── Stage 1: dependency installer ──────────────────────────────────────────
FROM python:3.14-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /install

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install/pkg -r requirements.txt

# ── Stage 2: model cache ────────────────────────────────────────────────────
FROM python:3.14-slim AS model-cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/model-cache

COPY --from=deps /install/pkg /usr/local

COPY scripts/cache_models.py ./scripts/cache_models.py
COPY src ./src
COPY setup.py ./

RUN python scripts/cache_models.py

# ── Stage 3: production runtime ─────────────────────────────────────────────
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /install/pkg /usr/local

# Copy pre-downloaded model cache
COPY --from=model-cache /model-cache /app/.cache

# Copy application source only (no dev tools, no test files)
COPY src ./src
COPY scripts ./scripts
COPY setup.py ./

# Install curl for health checks, then drop cache
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 app \
    && chown -R app:app /app
USER app

# Persisted data (vector store, eval results, golden dataset) lives here
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -sf http://localhost:8000/healthz || exit 1

ENTRYPOINT ["python"]
CMD ["-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
