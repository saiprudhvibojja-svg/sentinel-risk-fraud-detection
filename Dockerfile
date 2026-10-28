# ==============================================================================
# SentinelRisk Production Dockerfile
# Lean multi-stage build with non-root security and automated healthchecks
# ==============================================================================

# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Runtime Production Image
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SENTINEL_API_KEY=sentinel-dev-secret-key-2026 \
    ENVIRONMENT=production

WORKDIR /app

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder
COPY --from=builder /install /usr/local

# Create dedicated non-root application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Pre-create runtime directories and set ownership
RUN mkdir -p /app/logs /app/data/test_batches /app/models /app/reports /app/static && \
    chown -R appuser:appgroup /app

# Copy application source code and assets
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup static/ /app/static/
COPY --chown=appuser:appgroup models/ /app/models/
COPY --chown=appuser:appgroup reports/ /app/reports/
COPY --chown=appuser:appgroup scripts/ /app/scripts/
COPY --chown=appuser:appgroup pytest.ini /app/pytest.ini

# Switch to non-root execution
USER appuser

# Pre-generate sample test batches for instant batch auditing
RUN python /app/scripts/generate_test_batches.py

# Expose API port
EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production Entrypoint
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
